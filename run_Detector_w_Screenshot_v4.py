import atexit
import csv
import glob
import os
import torch
from Common_Files.Models import Detector
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

# v4 — active-learning helper:
#   PrintScreen  → silently save full screenshot (before Greenshot opens)
#   Home         → read clipboard crop, template-match → auto-add training entry

TRAINING_DIR = './Dataset/Sub_Set_v01/train/bjj/labeled'
CSV_PATH     = os.path.join(TRAINING_DIR, 'labels_bjj_combined.csv')
LABEL_NAME   = 'bjj'
MATCH_THRESH = 0.80   # minimum template-match confidence to accept

'''adding types to arguments'''
device = "cuda" if torch.cuda.is_available() else "cpu"

n_classes = 2
is_train  = False
model     = Detector(n_classes, device)
checkpoint = torch.load('./checkpoints/bjj_19' + '.pth')
model.load_state_dict(checkpoint['model_state'])
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

    full_pil.convert("RGB").save(dest_path, "JPEG", quality=95)

    with open(CSV_PATH, 'a', newline='') as f:
        csv.writer(f).writerow([LABEL_NAME, bx, by, bw, bh, new_name, img_w, img_h])

    print(f"Training sample saved: {new_name}  bbox=({bx},{by},{bw},{bh})")
    return new_name


# ── keyboard state ────────────────────────────────────────────────────────────
_last_full_screenshot = [None]   # set on PrintScreen, consumed on Home


def _commit_on_home():
    """
    Called when Home is pressed.
    Reads clipboard, template-matches against the saved screenshot, writes training entry.
    """
    if _last_full_screenshot[0] is None:
        print("No screenshot available — press PrintScreen first.")
        show_message(
            "⚠  No screenshot saved.\nPress PrintScreen before cropping.",
            duration=4,
            bg_color="#B71C1C",
            fg_color="white",
            window_width=380,
            window_height=80,
        )
        return

    clip_img = ImageGrab.grabclipboard()
    if clip_img is None:
        print("Clipboard does not contain an image.")
        show_message(
            "⚠  No image in clipboard.\nCrop with Greenshot first.",
            duration=4,
            bg_color="#B71C1C",
            fg_color="white",
            window_width=380,
            window_height=80,
        )
        return

    bbox = _find_crop_in_screenshot(clip_img, _last_full_screenshot[0])
    if bbox is None:
        show_message(
            "⚠  Could not locate crop in screenshot.\n"
            "Make sure you crop from the same frame.",
            duration=5,
            bg_color="#E65100",
            fg_color="white",
            window_width=420,
            window_height=90,
        )
        return

    name = _commit_training_sample(_last_full_screenshot[0], bbox)
    _last_full_screenshot[0] = None  # consume — avoid double-submit

    show_message(
        f"✓  Training sample added!\n{name}",
        duration=5,
        bg_color="#2E7D32",
        fg_color="white",
        window_width=380,
        window_height=80,
    )


def on_press(key):
    try:

        # ── INSERT: take screenshot and run detector ──────────────────────────
        if key == kb.Key.insert:
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
            _last_full_screenshot[0] = take_screenshot(monitor_index=1)
            print("Full screenshot captured for potential training use.")
            # silent — no toast, no interruption to the Greenshot workflow

        # ── HOME: commit training sample ─────────────────────────────────────
        elif key == kb.Key.home:
            threading.Thread(target=_commit_on_home, daemon=True).start()

        # ── ESC: quit ─────────────────────────────────────────────────────────
        if key == kb.Key.esc:
            print("Quitting...")
            _exit_info["reason"] = "User pressed ESC"
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
    print(f"Device: {device}")
    print("Press 'Insert'       to take a screenshot and detect objects.")
    print("Press 'PrintScreen'  to silently capture screen for training.")
    print("Press 'Home'         to add the Greenshot crop as a training sample.")
    print("Press 'ESC'          to quit.")

    show_message(
        "⌨  Insert: detect  |  ESC: quit\n"
        "PrintScreen → crop (Greenshot) → Home: add training sample",
        duration=7,
        bg_color="#1565C0",
        fg_color="white",
        window_width=460,
        window_height=90,
    )

    try:
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()
    except Exception as e:
        _exit_info["code"]   = 1
        _exit_info["reason"] = str(e)

    delete_results_folder()


if __name__ == "__main__":
    main()

