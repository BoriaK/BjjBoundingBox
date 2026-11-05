"""
run_Detector_on_Video.py

Reads a source video, runs the existing BJJ Faster R-CNN detector on it
(with a configurable detection cadence), tracks the detected ROI with a
smoothed, fixed-size crop window, and writes a cropped output video.

Example:
    python run_Detector_on_Video.py --input in.mp4 --output out.mp4
"""

import argparse

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as Ftv

from Common_Files.Models import Detector
from Common_Files.checkpoint_config import ACTIVE_CHECKPOINT

N_CLASSES = 2
IS_TRAIN = False


def parse_args():
    parser = argparse.ArgumentParser(description="Crop a video to the detected ROI using the BJJ detector.")
    parser.add_argument("--input", required=True, help="Path to the source video file.")
    parser.add_argument("--output", required=True, help="Path to write the cropped output video (.mp4).")
    parser.add_argument("--checkpoint", default=ACTIVE_CHECKPOINT,
                         help="Path to the detector checkpoint (default: %(default)s).")
    parser.add_argument("--detect-every-n", type=int, default=1,
                         help="Run detection every Nth frame; box is held between detections (default: %(default)s).")
    parser.add_argument("--smoothing-alpha", type=float, default=0.2,
                         help="EMA smoothing factor applied to the detected box center, 0-1 (default: %(default)s).")
    parser.add_argument("--margin", type=float, default=0.2,
                         help="Fractional padding added around the detected box when sizing the crop window (default: %(default)s).")
    parser.add_argument("--crop-width", type=int, default=None,
                         help="Fixed output crop width in pixels. If omitted, it is estimated via a pre-scan.")
    parser.add_argument("--crop-height", type=int, default=None,
                         help="Fixed output crop height in pixels. If omitted, it is estimated via a pre-scan.")
    parser.add_argument("--prescan-samples", type=int, default=30,
                         help="Number of evenly spaced frames sampled to estimate crop size (default: %(default)s).")
    return parser.parse_args()


def load_model(checkpoint_path, device):
    model = Detector(N_CLASSES, device)
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    print(f"Checkpoint epoch: {checkpoint.get('epoch')}")
    model.eval()
    return model


def detect_objects(model, device, frame_bgr):
    """Run the detector on a single BGR ndarray frame (as read by cv2)."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image_tensor = Ftv.to_tensor(frame_rgb)
    with torch.no_grad():
        predictions = model([image_tensor.to(device)], [], IS_TRAIN)
    return predictions


def process_predictions(predictions):
    """Pick the single highest-scoring box, mirroring run_Detector_w_Screenshot_v4.py."""
    scores_list = predictions[0]['scores'].tolist()
    if not scores_list:
        return None
    max_score_index = scores_list.index(max(scores_list))
    return predictions[0]['boxes'][max_score_index].tolist()


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def compute_crop_size(model, device, cap, frame_width, frame_height, frame_count, margin, sample_count):
    """Pre-scan evenly spaced frames to estimate a fixed crop size that fits the largest detected box."""
    max_w, max_h = 0.0, 0.0
    found_any = False

    sample_count = max(1, min(sample_count, frame_count if frame_count > 0 else sample_count))
    sample_indices = np.linspace(0, max(frame_count - 1, 0), sample_count, dtype=int)

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue

        box = process_predictions(detect_objects(model, device, frame))
        if box is None:
            continue

        found_any = True
        x1, y1, x2, y2 = box
        max_w = max(max_w, x2 - x1)
        max_h = max(max_h, y2 - y1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if not found_any:
        # No detections anywhere in the sample: fall back to a centered
        # crop covering most of the frame.
        return int(frame_width * 0.8), int(frame_height * 0.8)

    crop_w = int(max_w * (1.0 + margin))
    crop_h = int(max_h * (1.0 + margin))
    crop_w = min(crop_w, frame_width)
    crop_h = min(crop_h, frame_height)
    return max(crop_w, 1), max(crop_h, 1)


def clamped_crop_box(center, crop_w, crop_h, frame_width, frame_height):
    cx, cy = center
    x1 = cx - crop_w / 2.0
    y1 = cy - crop_h / 2.0

    x1 = max(0.0, min(x1, frame_width - crop_w))
    y1 = max(0.0, min(y1, frame_height - crop_h))

    return int(round(x1)), int(round(y1))


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.crop_width and args.crop_height:
        crop_w, crop_h = args.crop_width, args.crop_height
        print(f"Using fixed crop size from CLI: {crop_w}x{crop_h}")
    else:
        print(f"Pre-scanning up to {args.prescan_samples} frames to estimate crop size...")
        crop_w, crop_h = compute_crop_size(
            model, device, cap, frame_width, frame_height, frame_count,
            args.margin, args.prescan_samples,
        )
        print(f"Estimated crop size: {crop_w}x{crop_h}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (crop_w, crop_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output video for writing: {args.output}")

    smoothed_center = None
    frame_index = 0
    processed = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % args.detect_every_n == 0:
            box = process_predictions(detect_objects(model, device, frame))
            if box is not None:
                new_center = box_center(box)
                if smoothed_center is None:
                    smoothed_center = new_center
                else:
                    a = args.smoothing_alpha
                    smoothed_center = (
                        a * new_center[0] + (1 - a) * smoothed_center[0],
                        a * new_center[1] + (1 - a) * smoothed_center[1],
                    )

        if smoothed_center is None:
            # No detection yet at all: fall back to the frame center.
            smoothed_center = (frame_width / 2.0, frame_height / 2.0)

        x1, y1 = clamped_crop_box(smoothed_center, crop_w, crop_h, frame_width, frame_height)
        cropped = frame[y1:y1 + crop_h, x1:x1 + crop_w]
        writer.write(cropped)

        frame_index += 1
        processed += 1
        if processed % 100 == 0:
            print(f"Processed {processed}/{frame_count or '?'} frames...")

    cap.release()
    writer.release()
    print(f"Done. Wrote {processed} frames to {args.output}")


if __name__ == "__main__":
    main()
