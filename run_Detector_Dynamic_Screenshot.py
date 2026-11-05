"""
run_Detector_Dynamic_Screenshot.py

"Dynamic screenshot" — press 'Insert' to start recording the screen, press
'Insert' again (or wait for the max-duration cap) to stop. On stop, the
buffered clip is run through the BJJ detector, the ROI is tracked with a
smoothed fixed-size crop window, and a short cropped .gif is saved.

Press 'Ctrl+Q' at any time to exit the script (also stops an in-progress
recording without saving).
"""

import os
import time
import threading
from datetime import datetime

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as Ftv
import mss
from PIL import Image
import tkinter as tk
from pynput import keyboard as kb

from Common_Files.Models import Detector
from Common_Files.checkpoint_config import ACTIVE_CHECKPOINT

# ── configuration (tune freely) ───────────────────────────────────────────────
MAX_DURATION_SECONDS      = 30        # hard cap on a single recording
TARGET_FPS                = 15        # capture rate; experiment and adjust
CAPTURE_INTERVAL_SECONDS  = 1.0 / TARGET_FPS
OUTPUT_DIR                = r"C:\Users\Administrator\Videos\BJJ_Dynamic_Clips"
CHECKPOINT_PATH           = ACTIVE_CHECKPOINT
CROP_MARGIN               = 0.2       # fractional padding around detected boxes
SMOOTHING_ALPHA           = 0.2       # EMA factor for the tracked box center
MONITOR_INDEX             = 1

N_CLASSES = 2
IS_TRAIN  = False

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = Detector(N_CLASSES, device)
print(f"Loading checkpoint: {CHECKPOINT_PATH}")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state'])
print(f"Checkpoint epoch: {checkpoint.get('epoch')}")
model.eval()


# ── toast notifications (duplicated minimal pattern from v4) ─────────────────
def show_message(message, duration=4, bg_color="#4CAF50", fg_color="white",
                  window_width=420, window_height=90):
    """Display a small non-blocking toast that closes after *duration* seconds."""
    def create_and_show():
        root = tk.Tk()
        root.overrideredirect(True)
        root.attributes('-topmost', True)

        screen_width  = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x_position    = screen_width  - window_width  - 20
        y_position    = screen_height - window_height - 50
        root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

        frame = tk.Frame(root, bg=bg_color, borderwidth=2, relief="raised")
        frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        tk.Label(
            frame,
            text=message,
            font=("Arial", 12, "bold"),
            bg=bg_color,
            fg=fg_color,
            wraplength=window_width - 20,
        ).pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        root.after(duration * 1000, root.destroy)
        root.mainloop()

    threading.Thread(target=create_and_show, daemon=True).start()


def show_recording_toast(stop_event, message, start_time, bg_color="#B71C1C", fg_color="white",
                          window_width=520, window_height=90, poll_ms=150):
    """Persistent toast that stays visible until *stop_event* is set, showing
    a live count of elapsed recording seconds."""
    def create_and_show():
        root = tk.Tk()
        root.overrideredirect(True)
        root.attributes('-topmost', True)

        screen_width  = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x_position    = screen_width  - window_width  - 20
        y_position    = screen_height - window_height - 50
        root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

        frame = tk.Frame(root, bg=bg_color, borderwidth=2, relief="raised")
        frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        label = tk.Label(
            frame,
            text=message,
            font=("Arial", 12, "bold"),
            bg=bg_color,
            fg=fg_color,
            wraplength=window_width - 20,
        )
        label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        def check_stop():
            if stop_event.is_set():
                root.destroy()
            else:
                elapsed = time.time() - start_time
                label.config(text=f"{message}\n\u23F1  {elapsed:.1f}s")
                root.after(poll_ms, check_stop)

        root.after(poll_ms, check_stop)
        root.mainloop()

    threading.Thread(target=create_and_show, daemon=True).start()


# ── screen capture ────────────────────────────────────────────────────────────
def grab_frame_bgr(monitor_index=MONITOR_INDEX):
    """Grab one screen frame and return it as a BGR ndarray (cv2 convention)."""
    with mss.mss() as sct:
        monitors = sct.monitors
        monitor = monitors[monitor_index]
        # Match run_Detector_w_Screenshot_v4.py's take_screenshot(): on a
        # multi-monitor setup, monitor 1's reported size clips part of the
        # desktop the detector expects to see, so extend it the same way.
        if len(monitors) > 2 and monitor_index == 1:
            monitor = dict(monitor)
            monitor['width']  = monitor['width']  + 380
            monitor['height'] = monitor['height'] + 220
        shot = sct.grab(monitor)
        pil_image = Image.frombytes("RGB", shot.size, shot.rgb)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


# ── detection helpers (duplicated/adapted from run_Detector_on_Video.py) ─────
def detect_objects(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image_tensor = Ftv.to_tensor(frame_rgb)
    with torch.no_grad():
        predictions = model([image_tensor.to(device)], [], IS_TRAIN)
    return predictions


def process_predictions(predictions):
    scores_list = predictions[0]['scores'].tolist()
    if not scores_list:
        return None
    max_score_index = scores_list.index(max(scores_list))
    return predictions[0]['boxes'][max_score_index].tolist()


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def clamped_crop_box(center, crop_w, crop_h, frame_width, frame_height):
    cx, cy = center
    x1 = cx - crop_w / 2.0
    y1 = cy - crop_h / 2.0
    x1 = max(0.0, min(x1, frame_width - crop_w))
    y1 = max(0.0, min(y1, frame_height - crop_h))
    return int(round(x1)), int(round(y1))


# ── recording state ───────────────────────────────────────────────────────────
_state_lock = threading.Lock()
_recording_state = {
    "active": False,
    "frames": None,
    "start_time": None,
    "stop_event": None,
    "thread": None,
}


def _capture_loop(stop_event):
    frames = []
    start_time = time.time()

    while not stop_event.is_set():
        if time.time() - start_time >= MAX_DURATION_SECONDS:
            stop_event.set()
            break
        frame = grab_frame_bgr()
        frames.append(frame)
        stop_event.wait(CAPTURE_INTERVAL_SECONDS)

    elapsed = time.time() - start_time

    with _state_lock:
        _recording_state["frames"] = frames
        _recording_state["active"] = False

    _process_clip(frames, elapsed)


def _start_recording():
    with _state_lock:
        if _recording_state["active"]:
            return
        stop_event = threading.Event()
        start_time = time.time()
        _recording_state["active"]     = True
        _recording_state["frames"]     = None
        _recording_state["start_time"] = start_time
        _recording_state["stop_event"] = stop_event
        thread = threading.Thread(target=_capture_loop, args=(stop_event,), daemon=True)
        _recording_state["thread"] = thread

    thread.start()
    print(f"Recording started (max {MAX_DURATION_SECONDS}s)...")
    show_recording_toast(
        stop_event,
        f"\U0001F534  Recording... press Insert to stop\n(auto-stops after {MAX_DURATION_SECONDS}s)",
        start_time,
    )


def _stop_recording():
    with _state_lock:
        if not _recording_state["active"]:
            return
        stop_event = _recording_state["stop_event"]

    stop_event.set()
    print("Recording stopped. Processing clip...")
    show_message("\u23F3  Processing clip...", duration=3, bg_color="#546E7A")


# ── clip processing (reuses tracking logic from run_Detector_on_Video.py) ────
def _process_clip(frames, elapsed_seconds):
    if not frames:
        print("No frames captured — nothing to save.")
        show_message("\u26A0  No frames captured.", duration=4, bg_color="#B71C1C")
        return

    frame_height, frame_width = frames[0].shape[:2]
    fps = len(frames) / elapsed_seconds if elapsed_seconds > 0 else TARGET_FPS

    # Detect on every frame (clip is short — full detection is affordable).
    boxes = [process_predictions(detect_objects(f)) for f in frames]

    detected_boxes = [b for b in boxes if b is not None]
    print(f"Detected object in {len(detected_boxes)}/{len(boxes)} frames.")
    if detected_boxes:
        max_w = max(b[2] - b[0] for b in detected_boxes)
        max_h = max(b[3] - b[1] for b in detected_boxes)
        crop_w = min(int(max_w * (1.0 + CROP_MARGIN)), frame_width)
        crop_h = min(int(max_h * (1.0 + CROP_MARGIN)), frame_height)
        crop_w, crop_h = max(crop_w, 1), max(crop_h, 1)
    else:
        crop_w, crop_h = int(frame_width * 0.8), int(frame_height * 0.8)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"clip_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
    output_path = os.path.join(OUTPUT_DIR, filename)

    smoothed_center = None
    cropped_frames = []
    for frame, box in zip(frames, boxes):
        if box is not None:
            new_center = box_center(box)
            if smoothed_center is None:
                smoothed_center = new_center
            else:
                a = SMOOTHING_ALPHA
                smoothed_center = (
                    a * new_center[0] + (1 - a) * smoothed_center[0],
                    a * new_center[1] + (1 - a) * smoothed_center[1],
                )
        if smoothed_center is None:
            smoothed_center = (frame_width / 2.0, frame_height / 2.0)

        x1, y1 = clamped_crop_box(smoothed_center, crop_w, crop_h, frame_width, frame_height)
        cropped = frame[y1:y1 + crop_h, x1:x1 + crop_w]
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        cropped_frames.append(Image.fromarray(cropped_rgb))

    frame_duration_ms = int(round(1000.0 / fps)) if fps > 0 else int(round(1000.0 / TARGET_FPS))
    cropped_frames[0].save(
        output_path,
        save_all=True,
        append_images=cropped_frames[1:],
        duration=frame_duration_ms,
        loop=0,
    )
    print(f"Saved clip: {output_path}  ({len(frames)} frames, {fps:.1f} fps)")
    show_message(f"\u2713  Clip saved:\n{filename}", duration=5, bg_color="#2E7D32")


# ── keyboard listener ─────────────────────────────────────────────────────────
_CTRL_KEYS = (kb.Key.ctrl, kb.Key.ctrl_l, kb.Key.ctrl_r)
_modifier_state = {"ctrl": False}


def on_release(key):
    if key in _CTRL_KEYS:
        _modifier_state["ctrl"] = False


def on_press(key):
    if key in _CTRL_KEYS:
        _modifier_state["ctrl"] = True
        return

    try:
        # ── INSERT: toggle start/stop recording ───────────────────────────
        if key == kb.Key.insert:
            with _state_lock:
                is_active = _recording_state["active"]
            if is_active:
                _stop_recording()
            else:
                _start_recording()

        # ── CTRL+Q: stop the script cleanly ────────────────────────────────
        elif _modifier_state["ctrl"] and getattr(key, 'vk', None) == 0x51:
            print("Ctrl+Q pressed. Stopping...")
            with _state_lock:
                if _recording_state["active"]:
                    _recording_state["stop_event"].set()
            return False

    except AttributeError:
        pass


def main():
    print(f"Device: {device}")
    print("Press 'Insert'  to start recording, and again to stop; the clip is then detected, cropped, and saved as a .gif.")
    print(f"Recordings auto-stop after {MAX_DURATION_SECONDS} seconds.")
    print("Press 'Ctrl+Q'  to stop the script.")
    print(f"Clips are saved to: {OUTPUT_DIR}")

    show_message(
        "Press 'Insert'  to start recording, and again to stop.\n"
        "The clip is detected, cropped, and saved as a .gif.\n"
        f"Recordings auto-stop after {MAX_DURATION_SECONDS} seconds.\n"
        "Press 'Ctrl+Q'  to stop the script.",
        duration=7,
        bg_color="#1565C0",
        window_width=520,
        window_height=120,
    )

    try:
        with kb.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()
    except KeyboardInterrupt:
        print("Keyboard interrupt (Ctrl+C). Stopping...")
    except Exception as e:
        print(f"Stopped due to error: {e}")

    with _state_lock:
        if _recording_state["active"]:
            _recording_state["stop_event"].set()


if __name__ == "__main__":
    main()
