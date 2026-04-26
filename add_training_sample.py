"""
add_training_sample.py
======================
Interactive tool to annotate screenshots saved by the BJJ detector and add
them to the training dataset.

Workflow
--------
1. Run the main detector (main.py).
2. When nothing is detected the screenshot is auto-saved to ./Pending_Annotation/
   You can also press the **Home** key at any time to save the current screenshot.
3. Run THIS script:
       python add_training_sample.py
4. For each pending image an OpenCV window opens.
   • Click + drag  → draw bounding box around the BJJ technique.
   • ENTER / SPACE → confirm and save to training dataset.
   • R             → redraw (reset current box).
   • S             → skip this image (leave it pending).
   • ESC           → quit the annotator.
5. Annotated images are moved to ./Pending_Annotation/annotated/
6. Re-run run_Train_Detector.py (with --resume_epoch 19) to fine-tune the model.
"""

import csv
import glob
import os
import shutil

import cv2
import numpy as np
from PIL import Image

# ── paths ─────────────────────────────────────────────────────────────────────
PENDING_DIR  = './Pending_Annotation'
TRAINING_DIR = './Dataset/Sub_Set_v01/train/bjj/labeled'
CSV_PATH     = os.path.join(TRAINING_DIR, 'labels_bjj_combined.csv')
LABEL_NAME   = 'bjj'


# ── helpers ───────────────────────────────────────────────────────────────────

def next_image_name(folder: str) -> str:
    """Return the next available zero-padded 7-digit .jpg filename."""
    existing = glob.glob(os.path.join(folder, '*.jpg'))
    nums = []
    for p in existing:
        base = os.path.splitext(os.path.basename(p))[0]
        if base.isdigit():
            nums.append(int(base))
    next_num = (max(nums) + 1) if nums else 1
    return f'{next_num:07d}.jpg'


def append_csv(csv_path, label, bx, by, bw, bh, img_name, img_w, img_h):
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([label, bx, by, bw, bh, img_name, img_w, img_h])


# ── annotation window ─────────────────────────────────────────────────────────

class BBoxAnnotator:
    """OpenCV-based single-image bounding-box annotator."""

    HELP = ("Drag: draw box  |  ENTER/SPACE: save  |  R: reset  "
            "|  S: skip  |  ESC: quit")

    def __init__(self, image_path: str):
        self.image_path = image_path
        orig_bgr = cv2.imread(image_path)
        if orig_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        self.img_h, self.img_w = orig_bgr.shape[:2]

        # Scale down for display if the image is very large
        scale = min(1.0, 1400 / self.img_w, 860 / self.img_h)
        disp_w = int(self.img_w * scale)
        disp_h = int(self.img_h * scale)
        self.scale = scale
        self.display_base = cv2.resize(orig_bgr, (disp_w, disp_h))

        # drawing state
        self.drawing = False
        self.p1 = (-1, -1)
        self.p2 = (-1, -1)

    # ── mouse callback ────────────────────────────────────────────────────────

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.p1 = (x, y)
            self.p2 = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.p2 = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.p2 = (x, y)

    # ── draw helpers ──────────────────────────────────────────────────────────

    def _render(self) -> np.ndarray:
        frame = self.display_base.copy()
        if self.p1 != (-1, -1) and self.p2 != (-1, -1):
            cv2.rectangle(frame, self.p1, self.p2, (0, 255, 0), 2)
        # Help text – dark shadow then white
        for color, thickness in [((0, 0, 0), 3), ((255, 255, 255), 1)]:
            cv2.putText(frame, self.HELP, (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, thickness,
                        cv2.LINE_AA)
        return frame

    def _to_original(self, x, y):
        return int(round(x / self.scale)), int(round(y / self.scale))

    # ── public API ────────────────────────────────────────────────────────────

    def annotate(self):
        """
        Show the annotation window.

        Returns
        -------
        (bx, by, bw, bh)  – bounding box in original image pixel coords
        None              – user pressed S (skip)
        'quit'            – user pressed ESC
        """
        title = f"Annotate:  {os.path.basename(self.image_path)}"
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(title, self._mouse_cb)

        while True:
            cv2.imshow(title, self._render())
            key = cv2.waitKey(20) & 0xFF

            if key in (13, 32):  # Enter or Space → confirm
                if (self.p1 != (-1, -1) and
                        self.p2 != self.p1 and
                        abs(self.p2[0] - self.p1[0]) > 4 and
                        abs(self.p2[1] - self.p1[1]) > 4):
                    x0 = min(self.p1[0], self.p2[0])
                    y0 = min(self.p1[1], self.p2[1])
                    x1 = max(self.p1[0], self.p2[0])
                    y1 = max(self.p1[1], self.p2[1])
                    ox0, oy0 = self._to_original(x0, y0)
                    ox1, oy1 = self._to_original(x1, y1)
                    cv2.destroyWindow(title)
                    return ox0, oy0, ox1 - ox0, oy1 - oy0  # x, y, w, h
                else:
                    print("  ⚠  Draw a bounding box first (click and drag).")

            elif key in (ord('r'), ord('R')):   # Reset
                self.p1 = (-1, -1)
                self.p2 = (-1, -1)

            elif key in (ord('s'), ord('S')):   # Skip
                cv2.destroyWindow(title)
                return None

            elif key == 27:                     # ESC → quit
                cv2.destroyWindow(title)
                return 'quit'


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if not os.path.isdir(PENDING_DIR):
        print(f"No pending images folder found at:  {PENDING_DIR}")
        print("Save screenshots first:")
        print("  • The detector auto-saves a screenshot when nothing is detected.")
        print("  • Press 'Home' during detection to manually save the current screenshot.")
        return

    images = sorted(
        glob.glob(os.path.join(PENDING_DIR, '*.png')) +
        glob.glob(os.path.join(PENDING_DIR, '*.jpg'))
    )

    if not images:
        print("No images to annotate in Pending_Annotation/.")
        return

    print(f"Found {len(images)} image(s) to annotate.\n")
    done_dir = os.path.join(PENDING_DIR, 'annotated')
    os.makedirs(done_dir, exist_ok=True)

    saved = 0
    for img_path in images:
        print(f"  [{saved + 1}/{len(images)}]  {os.path.basename(img_path)}")
        try:
            annotator = BBoxAnnotator(img_path)
        except FileNotFoundError as e:
            print(f"  ✗ {e}")
            continue

        result = annotator.annotate()

        if result == 'quit':
            print("Quitting annotator.")
            break

        if result is None:
            print("  → Skipped.\n")
            continue

        bx, by, bw, bh = result

        # ── copy image to training folder ────────────────────────────────────
        new_name  = next_image_name(TRAINING_DIR)
        dest_path = os.path.join(TRAINING_DIR, new_name)
        pil_img   = Image.open(img_path).convert('RGB')
        img_w, img_h = pil_img.size
        pil_img.save(dest_path, 'JPEG', quality=95)

        # ── append CSV entry ─────────────────────────────────────────────────
        append_csv(CSV_PATH, LABEL_NAME, bx, by, bw, bh, new_name, img_w, img_h)

        print(f"  ✓  Saved as {new_name}  bbox=({bx}, {by}, {bw}, {bh})\n")

        # ── move source to annotated/ so it won't be re-processed ────────────
        shutil.move(img_path, os.path.join(done_dir, os.path.basename(img_path)))
        saved += 1

    print("─" * 60)
    print(f"Done.  {saved} image(s) added to the training dataset.")
    if saved:
        print("\nNext step – fine-tune the model from the last checkpoint:")
        print("  python run_Train_Detector.py --resume_epoch 19 --n_epochs 25")
        print("\nTip: use a small number of extra epochs (5-10) to avoid")
        print("     catastrophic forgetting of previously learned positions.")


if __name__ == '__main__':
    main()

