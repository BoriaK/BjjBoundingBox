import os
import torch
from Common_Files.Models import Detector
import cv2
import numpy as np
import torchvision.transforms.functional as Ftv
from PIL import Image
import mss
from pynput import keyboard as kb
import win32clipboard
import win32con
import io
import shutil
import psutil
import subprocess
import time

# Script Description:
# This script captures a screenshot from a selected monitor, uses a trained BJJ technique detector model
# to identify and draw bounding boxes around detected objects, and perform the following two actions:
# 2. Crop and copy the detected region to the clipboard.
# 1. Send the cropped detected region to Greenshot image editor.
# The script is intended for quick annotation and sharing of detected BJJ techniques from screen captures.

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

n_classes = 2
is_train = False
model = Detector(n_classes, device)
checkpoint = torch.load('./checkpoints/bjj_19.pth')
model.load_state_dict(checkpoint['model_state'])
model.eval()


def get_screen_info():
    with mss.mss() as sct:
        return sct.monitors


def take_screenshot(monitor_index=1):
    with mss.mss() as sct:
        monitors = sct.monitors
        if monitor_index >= len(monitors):
            raise ValueError(f"Monitor index {monitor_index} is out of range. Available monitors: {len(monitors) - 1}")

        monitor = monitors[monitor_index]
        if len(monitors) > 2 and monitor_index == 1:
            monitor['width'] += 380
            monitor['height'] += 220

        screenshot = sct.grab(monitor)
        return Image.frombytes("RGB", screenshot.size, screenshot.rgb)


def detect_objects(pil_image):
    image_tensor = Ftv.to_tensor(pil_image)
    with torch.no_grad():
        predictions = model([image_tensor.to(device)], [], is_train)
    return predictions


def process_predictions(predictions):
    boxes, labels, scores = [], [], []
    scores_list = predictions[0]['scores'].tolist()
    if scores_list:
        max_score_index = scores_list.index(max(scores_list))
        boxes.append(predictions[0]['boxes'][max_score_index].tolist())
        labels.append(predictions[0]['labels'][max_score_index].item())
        scores.append(predictions[0]['scores'][max_score_index].item())
    return boxes, labels, scores


def crop_and_save_image(pil_image, box, filename="cropped_image.png"):
    cropped_image = pil_image.crop((box[0], box[1], box[2], box[3]))
    dpi = 96
    target_cm_height = 4
    target_height_pixels = int((target_cm_height / 2.54) * dpi)
    aspect_ratio = cropped_image.width / cropped_image.height
    target_width_pixels = int(target_height_pixels * aspect_ratio)
    cropped_image_resized = cropped_image.resize((target_width_pixels, target_height_pixels))

    if not os.path.exists('./Results_for_Detector/'):
        os.makedirs('./Results_for_Detector/')
    cropped_image_resized.save('./Results_for_Detector/' + filename)
    print(f"Cropped and resized image saved as {filename} with height {target_cm_height} cm and proportional width")
    return cropped_image_resized


# def copy_image_to_clipboard(cropped_pil_image):
#     output = io.BytesIO()
#     cropped_pil_image.convert("RGB").save(output, "BMP")
#     data = output.getvalue()[14:]
#     output.close()
#
#     win32clipboard.OpenClipboard()
#     win32clipboard.EmptyClipboard()
#     win32clipboard.SetClipboardData(win32con.CF_DIB, data)
#     win32clipboard.CloseClipboard()
#     print("Cropped image copied to clipboard.")


def send_to_greenshot(pil_image, box):
    # Crop the image using the bounding box
    cropped_image = pil_image.crop((box[0], box[1], box[2], box[3]))

    # Convert the cropped image to BMP format and copy it to the clipboard
    output = io.BytesIO()
    cropped_image.save(output, "BMP")
    data = output.getvalue()[14:]
    output.close()

    win32clipboard.OpenClipboard()
    win32clipboard.EmptyClipboard()
    win32clipboard.SetClipboardData(win32con.CF_DIB, data)
    win32clipboard.CloseClipboard()
    print("Cropped image copied to clipboard.")

    # Check if Greenshot is already running
    greenshot_running = any("Greenshot.exe" in p.name() for p in psutil.process_iter())
    greenshot_path = r'"C:\Program Files\Greenshot\Greenshot.exe"'

    if greenshot_running:
        print("Greenshot is already running. Sending image to the editor...")
    else:
        print("Greenshot is not running. Launching Greenshot...")
        os.system(greenshot_path)
        # Wait a moment to ensure Greenshot is fully launched
        time.sleep(2)

    # Use the command-line argument to open the image in the editor
    try:
        subprocess.run(f'{greenshot_path} -openfromclipboard', shell=True, check=True)
        print("Image sent to Greenshot editor.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to send image to Greenshot: {e}")


def on_press(key):
    try:
        if key == kb.Key.insert:
            monitor_index = 1
            print(f"Taking screenshot of monitor {monitor_index}...")
            image = take_screenshot(monitor_index)
            print("Detecting objects...")
            predictions = detect_objects(image)
            boxes, labels, scores = process_predictions(predictions)
            if not boxes:
                print("No objects detected. Quitting...")
                return False

            print(f"Detected {len(boxes)} objects:")
            for i in range(len(boxes)):
                print(f"Object {i + 1}:")
                print(f"  Bounding Box: {boxes[i]}")
                print(f"  Label: {labels[i]}")
                print(f"  Score: {scores[i]:.4f}")

            # Always send the cropped image to Greenshot
            send_to_greenshot(image, boxes[0])

        if key == kb.Key.esc:
            print("Quitting...")
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
    print("Press 'Insert' to take a screenshot and detect objects.")
    print("Press 'ESC' to quit.")

    with kb.Listener(on_press=on_press) as listener:
        listener.join()

    delete_results_folder()


if __name__ == "__main__":
    main()