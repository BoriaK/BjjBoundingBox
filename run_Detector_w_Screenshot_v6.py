import atexit
import csv
import glob
import hashlib
import os
import time
from datetime import datetime
import torch
from Common_Files.Models import Detector
from Common_Files.checkpoint_config import ACTIVE_CHECKPOINT
import cv2
import numpy as np
import torchvision.transforms.functional as Ftv
from PIL import Image, ImageGrab
import mss
from pynput import keyboard
import win32clipboard
import win32con
import io
from pynput import keyboard as kb
import shutil
import tkinter as tk
import threading

# v6 — same as v5, but the dynamic-recording flow can skip the bbox detector
# entirely (DYNAMIC_RUN_BBOX_DETECTOR below) and save the original,
# uncropped frames instead of a tracked/cropped clip:
#   Insert         → classical: screenshot + detect + crop + copy to clipboard
#   Shift+Insert   → dynamic: start/stop a short recording, then crop+save as .gif
#   PrintScreen    → silently save full screenshot and wait for Greenshot crop
#   Home           → undo the last auto-saved training entry
#   Ctrl+Q         → stop the script

TRAINING_DIR = './Dataset/Sub_Set_v01/train/bjj/labeled'
CSV_PATH     = os.path.join(TRAINING_DIR, 'labels_bjj_combined.csv')
LABEL_NAME   = 'bjj'
MATCH_THRESH = 0.80   # minimum template-match confidence to accept
AUTO_COMMIT_TIMEOUT = 45
AUTO_COMMIT_POLL_SECONDS = 0.25

# ── dynamic-screenshot configuration (tune freely) ────────────────────────────
DYNAMIC_MAX_DURATION_SECONDS = 20        # hard cap on a single recording
DYNAMIC_TARGET_FPS           = 15        # capture rate; experiment and adjust
DYNAMIC_CAPTURE_INTERVAL     = 1.0 / DYNAMIC_TARGET_FPS
DYNAMIC_OUTPUT_DIR           = r"C:\Users\Administrator\Videos\BJJ_Dynamic_Clips"
DYNAMIC_CROP_MARGIN          = 0.2       # fractional padding around detected boxes
DYNAMIC_SMOOTHING_ALPHA      = 0.2       # EMA factor for the tracked box center

# When True, each captured frame is run through the bbox detector and the
# clip is cropped/tracked around the detected object (original v5 behavior).
# When False, the detector is skipped entirely and the full, uncropped
# frames are saved as-is — i.e. the original video is kept untouched.
DYNAMIC_RUN_BBOX_DETECTOR   = True

'''adding types to arguments'''
device = "cuda" if torch.cuda.is_available() else "cpu"

n_classes = 2
is_train  = False
model     = Detector(n_classes, device)
print(f"Loading checkpoint: {ACTIVE_CHECKPOINT}")
checkpoint = torch.load(ACTIVE_CHECKPOINT)
model.load_state_dict(checkpoint['model_state'])
print(f"Checkpoint epoch: {checkpoint.get('epoch')}")
model.eval()

# ── exit notification ─────────────────────────────────────────────────────────
_exit_info = {"code": 0, "reason": ""}


def _show_exit_notification(message, duration=5):
    """Blocking yellow toast shown when the script exits (runs via atexit)."""
    try:
        root = tk.Tk()
        root.overrideredirect(True)
        root.attributes('-topmost', True)

        screen_width  = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        window_width  = 500
        window_height = 70
        x = screen_width  - window_width  - 20
        y = screen_height - window_height - 50
        root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        frame = tk.Frame(root, bg="#FFD700", borderwidth=2, relief="raised")
        frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        tk.Label(
            frame,
            text=message,
            font=("Arial", 11, "bold"),
            bg="#FFD700",
            fg="#333333",
            wraplength=window_width - 20,
        ).pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        root.after(duration * 1000, root.destroy)
        root.mainloop()
    except Exception:
        pass


def _on_exit():
    code   = _exit_info["code"]
    reason = _exit_info["reason"]
    msg    = f"Process finished with exit code {code}"
    if reason:
        msg += f"  —  {reason}"
    _show_exit_notification(msg)


atexit.register(_on_exit)
# ─────────────────────────────────────────────────────────────────────────────


def get_screen_info():
    with mss.mss() as sct:
        return sct.monitors


def take_screenshot(monitor_index=1):
    with mss.mss() as sct:
        monitors = sct.monitors
        if monitor_index >= len(monitors):
            raise ValueError(
                f"Monitor index {monitor_index} is out of range. "
                f"Available monitors: {len(monitors) - 1}"
            )
        monitor = monitors[monitor_index]
        if len(monitors) > 2 and monitor_index == 1:
            monitor['width']  = monitor['width']  + 380
            monitor['height'] = monitor['height'] + 220

        screenshot     = sct.grab(monitor)
        screenshot_pil = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
    return screenshot_pil


def detect_objects(pil_image):
    image_tensor = Ftv.to_tensor(pil_image)
    with torch.no_grad():
        predictions = model([image_tensor.to(device)], [], is_train)
    return predictions


def process_predictions(predictions):
    boxes  = []
    labels = []
    scores = []

    scores_list = predictions[0]['scores'].tolist()
    if scores_list:
        max_score_index = scores_list.index(max(scores_list))
        boxes.append(predictions[0]['boxes'][max_score_index].tolist())
        labels.append(predictions[0]['labels'][max_score_index].item())
        scores.append(predictions[0]['scores'][max_score_index].item())

    return boxes, labels, scores


def draw_boxes(pil_image, boxes):
    for b in boxes:
        start_point = (int(np.round(b[0])), int(np.round(b[1])))
        end_point   = (int(np.round(b[2])), int(np.round(b[3])))
        color       = (255, 0, 0)
        thickness   = 2
        im_np_array = np.array(pil_image)
        im_np_array = cv2.cvtColor(im_np_array, cv2.COLOR_BGR2RGB)
        im          = cv2.rectangle(im_np_array, start_point, end_point, color, thickness)
    return im


def crop_and_save_image(pil_image, box, filename="cropped_image.png"):
    cropped_image = pil_image.crop((box[0], box[1], box[2], box[3]))

    dpi                  = 96
    target_cm_height     = 4
    target_height_pixels = int((target_cm_height / 2.54) * dpi)
    aspect_ratio         = cropped_image.width / cropped_image.height
    target_width_pixels  = int(target_height_pixels * aspect_ratio)

    cropped_image_resized = cropped_image.resize((target_width_pixels, target_height_pixels))

    if not os.path.exists('./Results_for_Detector/'):
        os.makedirs('./Results_for_Detector/')

    cropped_image_resized.save('./Results_for_Detector/' + filename)
    print(f"Cropped and resized image saved as {filename} "
          f"with height {target_cm_height} cm and proportional width")
    return cropped_image_resized


def copy_image_to_clipboard(cropped_pil_image):
    output = io.BytesIO()
    cropped_pil_image.convert("RGB").save(output, "BMP")
    data = output.getvalue()[14:]
    output.close()

    win32clipboard.OpenClipboard()
    win32clipboard.EmptyClipboard()
    win32clipboard.SetClipboardData(win32con.CF_DIB, data)
    win32clipboard.CloseClipboard()
    print("Cropped image copied to clipboard.")


def show_message(message, duration=5, bg_color="#4CAF50", fg_color="white",
                 window_width=400, window_height=80):
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


def show_recording_toast(stop_event, message, bg_color="#B71C1C", fg_color="white",
                          window_width=520, window_height=90, poll_ms=150):
    """Persistent toast that stays visible until *stop_event* is set."""
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

        def check_stop():
            if stop_event.is_set():
                root.destroy()
            else:
                root.after(poll_ms, check_stop)

        root.after(poll_ms, check_stop)
        root.mainloop()

    threading.Thread(target=create_and_show, daemon=True).start()


# ── training-data helpers ─────────────────────────────────────────────────────

def _next_image_name():
    """Return the next available zero-padded 7-digit .jpg filename."""
    existing = glob.glob(os.path.join(TRAINING_DIR, '*.jpg'))
    nums = [int(os.path.splitext(os.path.basename(p))[0])
            for p in existing
            if os.path.splitext(os.path.basename(p))[0].isdigit()]
    return f'{(max(nums) + 1) if nums else 1:07d}.jpg'


def _find_crop_in_screenshot(crop_pil, full_pil):
    """
    Template-match *crop_pil* inside *full_pil*.
    Returns (x, y, w, h) in full-image pixel coords, or None if not found.
    """
    full_bgr = cv2.cvtColor(np.array(full_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    crop_bgr = cv2.cvtColor(np.array(crop_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

    fh, fw = full_bgr.shape[:2]
    ch, cw = crop_bgr.shape[:2]

    if ch > fh or cw > fw:
        print("Crop is larger than the saved screenshot — cannot match.")
        return None

    # match on grayscale for robustness
    full_gray = cv2.cvtColor(full_bgr, cv2.COLOR_BGR2GRAY)
    crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(full_gray, crop_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    print(f"Template match confidence: {max_val:.3f}")
    if max_val < MATCH_THRESH:
        print(f"Match confidence {max_val:.3f} below threshold {MATCH_THRESH} — skipping.")
        return None

    x, y = max_loc
    return x, y, cw, ch   # bbox as x, y, width, height


def _commit_training_sample(full_pil, bbox):
    """Save full_pil to the training folder and append a CSV row."""
    bx, by, bw, bh = bbox
    img_w, img_h   = full_pil.size
    new_name       = _next_image_name()
    dest_path      = os.path.join(TRAINING_DIR, new_name)

    os.makedirs(TRAINING_DIR, exist_ok=True)

    full_pil.convert("RGB").save(dest_path, "JPEG", quality=95)

    with open(CSV_PATH, 'a', newline='') as f:
        csv.writer(f).writerow([LABEL_NAME, bx, by, bw, bh, new_name, img_w, img_h])

    print(f"Training sample saved: {new_name}  bbox=({bx},{by},{bw},{bh})")
    return new_name, dest_path


# ── keyboard state ────────────────────────────────────────────────────────────
_pending_capture = {
    "full_screenshot": None,
    "clipboard_signature": None,
    "token": 0,
}
_last_saved_sample = {
    "image_name": None,
    "image_path": None,
}
_state_lock = threading.Lock()


def _get_clipboard_image():
    clip_obj = ImageGrab.grabclipboard()
    if isinstance(clip_obj, Image.Image):
        return clip_obj.convert("RGB")
    return None


def _image_signature(pil_image):
    rgb_image = pil_image.convert("RGB")
    return f"{rgb_image.size[0]}x{rgb_image.size[1]}:{hashlib.md5(rgb_image.tobytes()).hexdigest()}"


def _clear_pending_capture(token=None):
    with _state_lock:
        if token is not None and _pending_capture["token"] != token:
            return False
        _pending_capture["full_screenshot"] = None
        _pending_capture["clipboard_signature"] = None
        if token is None:
            _pending_capture["token"] += 1
        return True


def _queue_capture_for_auto_commit(full_screenshot):
    baseline_clip = _get_clipboard_image()
    baseline_signature = _image_signature(baseline_clip) if baseline_clip is not None else None

    with _state_lock:
        _pending_capture["token"] += 1
        token = _pending_capture["token"]
        _pending_capture["full_screenshot"] = full_screenshot
        _pending_capture["clipboard_signature"] = baseline_signature

    threading.Thread(target=_wait_for_crop_and_commit, args=(token,), daemon=True).start()


def _wait_for_crop_and_commit(token):
    attempted_signatures = set()
    started_at = time.time()

    while time.time() - started_at < AUTO_COMMIT_TIMEOUT:
        with _state_lock:
            if _pending_capture["token"] != token:
                return
            full_screenshot = _pending_capture["full_screenshot"]
            baseline_signature = _pending_capture["clipboard_signature"]

        clip_img = _get_clipboard_image()
        if clip_img is None:
            time.sleep(AUTO_COMMIT_POLL_SECONDS)
            continue

        clip_signature = _image_signature(clip_img)
        if clip_signature == baseline_signature or clip_signature in attempted_signatures:
            time.sleep(AUTO_COMMIT_POLL_SECONDS)
            continue

        attempted_signatures.add(clip_signature)
        bbox = _find_crop_in_screenshot(clip_img, full_screenshot)
        if bbox is None:
            time.sleep(AUTO_COMMIT_POLL_SECONDS)
            continue

        result = _commit_training_sample(full_screenshot, bbox)
        image_name, image_path = result

        with _state_lock:
            if _pending_capture["token"] != token:
                return
            _pending_capture["full_screenshot"] = None
            _pending_capture["clipboard_signature"] = None
            _last_saved_sample["image_name"] = image_name
            _last_saved_sample["image_path"] = image_path

        show_message(
            f"✓  Training sample added automatically!\n{image_name}",
            duration=5,
            bg_color="#2E7D32",
            fg_color="white",
            window_width=420,
            window_height=80,
        )
        return

    if _clear_pending_capture(token):
        print("Timed out waiting for the Greenshot crop to appear in the clipboard.")
        show_message(
            "⚠  Timed out waiting for the Greenshot crop.\nPress PrintScreen and try again.",
            duration=5,
            bg_color="#E65100",
            fg_color="white",
            window_width=430,
            window_height=90,
        )


def _undo_last_training_sample():
    with _state_lock:
        image_name = _last_saved_sample["image_name"]
        image_path = _last_saved_sample["image_path"]

    if not image_name or not image_path:
        print("No auto-saved training sample is available to undo for the current capture cycle.")
        show_message(
            "⚠  Dataset was not updated in this capture cycle.\nNothing to remove.",
            duration=4,
            bg_color="#B71C1C",
            fg_color="white",
            window_width=470,
            window_height=85,
        )
        return

    if os.path.exists(image_path):
        os.remove(image_path)
        print(f"Removed training image: {image_path}")
    else:
        print(f"Training image already missing: {image_path}")

    rows = []
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'r', newline='') as f:
            rows = list(csv.reader(f))

        rows = [row for row in rows if len(row) < 6 or row[5] != image_name]

        with open(CSV_PATH, 'w', newline='') as f:
            csv.writer(f).writerows(rows)
        print(f"Removed annotation row for: {image_name}")

    with _state_lock:
        _last_saved_sample["image_name"] = None
        _last_saved_sample["image_path"] = None

    show_message(
        f"↶  Removed last training sample\n{image_name}",
        duration=5,
        bg_color="#6A1B9A",
        fg_color="white",
        window_width=400,
        window_height=80,
    )


# ── dynamic-screenshot state and helpers ──────────────────────────────────────
_dynamic_lock  = threading.Lock()
_dynamic_state = {
    "active": False,
    "stop_event": None,
    "thread": None,
}


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


def _dynamic_capture_loop(stop_event):
    frames = []
    start_time = time.time()

    while not stop_event.is_set():
        if time.time() - start_time >= DYNAMIC_MAX_DURATION_SECONDS:
            stop_event.set()
            break
        frames.append(take_screenshot(monitor_index=1))
        stop_event.wait(DYNAMIC_CAPTURE_INTERVAL)

    elapsed = time.time() - start_time

    with _dynamic_lock:
        _dynamic_state["active"] = False

    _process_dynamic_clip(frames, elapsed)


def _start_dynamic_recording():
    with _dynamic_lock:
        if _dynamic_state["active"]:
            return
        stop_event = threading.Event()
        _dynamic_state["active"]     = True
        _dynamic_state["stop_event"] = stop_event
        thread = threading.Thread(target=_dynamic_capture_loop, args=(stop_event,), daemon=True)
        _dynamic_state["thread"] = thread

    thread.start()
    print(f"(Shift+Insert) Dynamic recording started (max {DYNAMIC_MAX_DURATION_SECONDS}s)...")
    show_recording_toast(
        stop_event,
        f"\U0001F534  Recording... press Shift+Insert to stop\n"
        f"(auto-stops after {DYNAMIC_MAX_DURATION_SECONDS}s)",
    )


def _stop_dynamic_recording():
    with _dynamic_lock:
        if not _dynamic_state["active"]:
            return
        stop_event = _dynamic_state["stop_event"]

    stop_event.set()
    print("Dynamic recording stopped. Processing clip...")
    show_message("\u23F3  Processing clip...", duration=3, bg_color="#546E7A")


def _process_dynamic_clip(frames, elapsed_seconds):
    if not frames:
        print("No frames captured — nothing to save.")
        show_message("\u26A0  No frames captured.", duration=4, bg_color="#B71C1C")
        return

    frame_width, frame_height = frames[0].size
    fps = len(frames) / elapsed_seconds if elapsed_seconds > 0 else DYNAMIC_TARGET_FPS

    os.makedirs(DYNAMIC_OUTPUT_DIR, exist_ok=True)
    filename = f"clip_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
    output_path = os.path.join(DYNAMIC_OUTPUT_DIR, filename)

    if DYNAMIC_RUN_BBOX_DETECTOR:
        # Detect on every frame (clip is short — full detection is affordable).
        boxes = []
        for frame in frames:
            detected_boxes, _, _ = process_predictions(detect_objects(frame))
            boxes.append(detected_boxes[0] if detected_boxes else None)

        detected_boxes = [b for b in boxes if b is not None]
        print(f"Detected object in {len(detected_boxes)}/{len(boxes)} frames.")
        if detected_boxes:
            max_w = max(b[2] - b[0] for b in detected_boxes)
            max_h = max(b[3] - b[1] for b in detected_boxes)
            crop_w = min(int(max_w * (1.0 + DYNAMIC_CROP_MARGIN)), frame_width)
            crop_h = min(int(max_h * (1.0 + DYNAMIC_CROP_MARGIN)), frame_height)
            crop_w, crop_h = max(crop_w, 1), max(crop_h, 1)
        else:
            crop_w, crop_h = int(frame_width * 0.8), int(frame_height * 0.8)

        smoothed_center = None
        cropped_frames = []
        for frame, box in zip(frames, boxes):
            if box is not None:
                new_center = box_center(box)
                if smoothed_center is None:
                    smoothed_center = new_center
                else:
                    a = DYNAMIC_SMOOTHING_ALPHA
                    smoothed_center = (
                        a * new_center[0] + (1 - a) * smoothed_center[0],
                        a * new_center[1] + (1 - a) * smoothed_center[1],
                    )
            if smoothed_center is None:
                smoothed_center = (frame_width / 2.0, frame_height / 2.0)

            x1, y1 = clamped_crop_box(smoothed_center, crop_w, crop_h, frame_width, frame_height)
            cropped_frames.append(frame.crop((x1, y1, x1 + crop_w, y1 + crop_h)))
    else:
        # Detector disabled — keep the original, full-resolution frames untouched.
        print("Bbox detector disabled for this clip — saving original frames as-is.")
        cropped_frames = frames

    frame_duration_ms = int(round(1000.0 / fps)) if fps > 0 else int(round(1000.0 / DYNAMIC_TARGET_FPS))
    cropped_frames[0].save(
        output_path,
        save_all=True,
        append_images=cropped_frames[1:],
        duration=frame_duration_ms,
        loop=0,
    )
    print(f"Saved clip: {output_path}  ({len(frames)} frames, {fps:.1f} fps)")
    show_message(f"\u2713  Clip saved:\n{filename}", duration=5, bg_color="#2E7D32")


_CTRL_KEYS  = (kb.Key.ctrl, kb.Key.ctrl_l, kb.Key.ctrl_r)
_SHIFT_KEYS = (kb.Key.shift, kb.Key.shift_l, kb.Key.shift_r)
_modifier_state = {"ctrl": False, "shift": False}


def on_release(key):
    if key in _CTRL_KEYS:
        _modifier_state["ctrl"] = False
    elif key in _SHIFT_KEYS:
        _modifier_state["shift"] = False


def on_press(key):
    if key in _CTRL_KEYS:
        _modifier_state["ctrl"] = True
        return
    if key in _SHIFT_KEYS:
        _modifier_state["shift"] = True
        return

    try:

        # ── INSERT: classical flow, or dynamic recording if Shift is held ────
        if key == kb.Key.insert:
            if _modifier_state["shift"]:
                with _dynamic_lock:
                    is_active = _dynamic_state["active"]
                if is_active:
                    _stop_dynamic_recording()
                else:
                    _start_dynamic_recording()
                return

            monitor_index = 1
            print(f"Taking screenshot of monitor {monitor_index}...")
            image = take_screenshot(monitor_index)

            print("Detecting objects...")
            predictions            = detect_objects(image)
            boxes, labels, scores  = process_predictions(predictions)

            if not boxes:
                print("No objects detected. Quitting...")
                _exit_info["reason"] = "No objects detected"
                return False

            print(f"Detected {len(boxes)} objects:")
            for i in range(len(boxes)):
                print(f"Object {i + 1}:")
                print(f"  Bounding Box: {boxes[i]}")
                print(f"  Label:        {labels[i]}")
                print(f"  Score:        {scores[i]:.4f}")

            draw_boxes(image, boxes)
            cropped_image = crop_and_save_image(image, boxes[0])
            copy_image_to_clipboard(cropped_image)
            show_message("✓ Screenshot saved and copied to clipboard!", duration=5)

        # ── PRINT SCREEN: silently snapshot the screen for training ──────────
        elif key == kb.Key.print_screen:
            # Capture immediately — before Greenshot's overlay appears.
            # Do NOT return False so Greenshot still receives the key.
            full_screenshot = take_screenshot(monitor_index=1)
            # New capture cycle starts here: Home should only undo a sample
            # created from this cycle, never an older one.
            with _state_lock:
                _last_saved_sample["image_name"] = None
                _last_saved_sample["image_path"] = None
            _queue_capture_for_auto_commit(full_screenshot)
            print("Full screenshot captured for potential training use.")
            show_message(
                "📷  Screenshot captured.\nFinish the Greenshot crop to auto-add training data.",
                duration=3,
                bg_color="#546E7A",
                fg_color="white",
                window_width=560,
                window_height=100,
            )

        # ── HOME: undo last auto-saved training sample ───────────────────────
        elif key == kb.Key.home:
            threading.Thread(target=_undo_last_training_sample, daemon=True).start()

        # ── CTRL+Q: stop the script cleanly ───────────────────────────
        # Note: with Ctrl held, key.char is often a control character (not
        # 'q'), so match on the physical key's virtual-key code instead.
        elif _modifier_state["ctrl"] and getattr(key, 'vk', None) == 0x51:
            print("Ctrl+Q pressed. Stopping...")
            with _dynamic_lock:
                if _dynamic_state["active"]:
                    _dynamic_state["stop_event"].set()
            _exit_info["reason"] = "User requested exit (Ctrl+Q)"
            return False

    except AttributeError:
        pass


def delete_results_folder(folder_path='./Results_for_Detector/'):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' and all its contents have been deleted.")
    else:
        print(f"Folder '{folder_path}' does not exist.")


def main():
    dynamic_mode_desc = (
        "tracked+cropped" if DYNAMIC_RUN_BBOX_DETECTOR else "original, uncropped"
    )

    print(f"Device: {device}")
    print("Press 'Insert'       to screenshot, detect, crop to the bounding box, and copy the crop to the clipboard.")
    print(f"Press 'Shift+Insert' to start/stop a dynamic-screenshot recording (max {DYNAMIC_MAX_DURATION_SECONDS}s), "
          f"saved as a {dynamic_mode_desc} .gif.")
    print("Press 'PrintScreen'  to silently capture the full screen; finishing a Greenshot crop auto-adds a training sample.")
    print("Press 'Home'         to remove the last auto-saved training sample.")
    print("Press 'Ctrl+Q'       to stop the script.")
    print(f"DYNAMIC_RUN_BBOX_DETECTOR = {DYNAMIC_RUN_BBOX_DETECTOR}  "
          f"(set to False in the script to keep the original video without running the detector)")

    show_message(
        "Press 'Insert'       to screenshot, detect, crop, and copy the crop to clipboard.\n"
        f"Press 'Shift+Insert' to start/stop a dynamic recording, saved as a {dynamic_mode_desc} .gif.\n"
        "Press 'PrintScreen'  to silently capture screen for automatic training.\n"
        "Press 'Home'         to remove the last auto-saved training sample.\n"
        "Press 'Ctrl+Q'       to stop the script.",
        duration=7,
        bg_color="#1565C0",
        fg_color="white",
        window_width=560,
        window_height=160,
    )

    try:
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()
    except KeyboardInterrupt:
        _exit_info["reason"] = _exit_info["reason"] or "Keyboard interrupt (Ctrl+C)"
    except Exception as e:
        _exit_info["code"]   = 1
        _exit_info["reason"] = str(e)

    with _dynamic_lock:
        if _dynamic_state["active"]:
            _dynamic_state["stop_event"].set()

    delete_results_folder()


if __name__ == "__main__":
    main()
