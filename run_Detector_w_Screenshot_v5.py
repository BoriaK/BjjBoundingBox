import os
import torch
from Common_Files.Models import Detector
import cv2
import numpy as np
import torchvision.transforms.functional as Ftv
from PIL import Image, ImageOps
import mss
from pynput import keyboard
import win32clipboard
import win32con
import io
from pynput import keyboard as kb
import shutil
import psutil
import subprocess
import time

# This script is testing the detector, to detect BJJ technique and place bounding box around them

'''adding types to arguments'''
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

n_classes = 2
is_train = False
model = Detector(n_classes, device)
if device == "cpu":
    checkpoint = torch.load('./checkpoints/bjj_19' + '.pth')
else:
    checkpoint = torch.load('./checkpoints/bjj_19' + '.pth')
model.load_state_dict(checkpoint['model_state'])
model.eval()


# ts = time.time()
# root = r'./Dataset/Sub_Set_v01'

# Get all monitors information
def get_screen_info():
    with mss.mss() as sct:
        monitors = sct.monitors
        # for i, monitor in enumerate(monitors):
        #     print(
        #         f"Monitor: {i}, Left: {monitor['left']}, Top: {monitor['top']}, Width: {monitor['width']}, Height: {monitor['height']}")
        return monitors


def take_screenshot(monitor_index=1):
    with mss.mss() as sct:
        monitors = sct.monitors
        if monitor_index >= len(monitors):
            raise ValueError(f"Monitor index {monitor_index} is out of range. Available monitors: {len(monitors) - 1}")

        monitor = monitors[monitor_index]
        # a correction to the partial screenshot on the small monitor
        if len(monitors) > 2 and monitor_index == 1:
            monitor['width'] = monitor['width'] + 380
            monitor['height'] = monitor['height'] + 220

        # Take a screenshot
        screenshot = sct.grab(monitor)
        # Convert the screenshot to a PIL image
        screenshot_pil = Image.frombytes("RGB", screenshot.size, screenshot.rgb)

        # for debug:
        # screenshot_pil.show(title="Screenshot")
        ##################################
        # screenshot_np = np.array(screenshot)

    return screenshot_pil


def detect_objects(pil_image):
    # Convert the NumPy array back to a PIL Image for transformation
    # image_pil = Image.fromarray(image)

    # Apply the transformations to the image
    image_tensor = Ftv.to_tensor(pil_image)

    # Add a batch dimension
    # image_tensor = image_tensor.unsqueeze(0)

    # Perform the object detection
    with torch.no_grad():
        predictions = model([image_tensor.to(device)], [], is_train)

    return predictions


def process_predictions(predictions):
    boxes = []
    labels = []
    scores = []

    # Get the scores and find the index of the highest score
    scores_list = predictions[0]['scores'].tolist()
    if scores_list:
        max_score_index = scores_list.index(max(scores_list))

        # Append only the highest scoring prediction
        boxes.append(predictions[0]['boxes'][max_score_index].tolist())
        labels.append(predictions[0]['labels'][max_score_index].item())
        scores.append(predictions[0]['scores'][max_score_index].item())

    return boxes, labels, scores


def draw_boxes(pil_image, boxes):
    for b in boxes:
        start_point = (int(np.round(b[0])), int(np.round(b[1])))
        end_point = (int(np.round(b[2])), int(np.round(b[3])))
        color = (255, 0, 0)  # Red color for the rectangle
        thickness = 2
        im_np_array = np.array(pil_image)
        # Convert BGR back to RGB
        im_np_array = cv2.cvtColor(im_np_array, cv2.COLOR_BGR2RGB)
        im = cv2.rectangle(im_np_array, start_point, end_point, color, thickness)

    return im


def crop_and_save_image(pil_image, box, filename="cropped_image.png"):
    # Crop the image using the bounding box
    cropped_image = pil_image.crop((box[0], box[1], box[2], box[3]))

    # Convert cm to pixels (assuming 96 DPI)
    dpi = 96  # standard screen DPI
    target_cm_height = 4  # resize height to 4 cm
    target_height_pixels = int((target_cm_height / 2.54) * dpi)  # Convert 4 cm to pixels

    # Get the aspect ratio of the cropped image
    aspect_ratio = cropped_image.width / cropped_image.height

    # Calculate the new width while maintaining the aspect ratio
    target_width_pixels = int(target_height_pixels * aspect_ratio)

    # Resize the image to the new width and height (keeping proportional width)
    cropped_image_resized = cropped_image.resize((target_width_pixels, target_height_pixels))

    # Ensure the folder exists, or create it
    if not os.path.exists('./Results_for_Detector/'):
        os.makedirs('./Results_for_Detector/')

    # Save the resized image
    cropped_image_resized.save('./Results_for_Detector/' + filename)
    print(f"Cropped and resized image saved as {filename} with height {target_cm_height} cm and proportional width")

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
    print("Cropped image copied to clipboard (for Greenshot).")

    # Check if Greenshot is already running
    greenshot_running = any("Greenshot.exe" in p.name() for p in psutil.process_iter())
    greenshot_path = r'"C:\Program Files\Greenshot\Greenshot.exe"'

    if greenshot_running:
        print("Greenshot is already running. Sending image to the editor...")
    else:
        print("Greenshot is not running. Launching Greenshot...")
        os.system(greenshot_path)
        time.sleep(2)

    try:
        subprocess.run(f'{greenshot_path} -openfromclipboard', shell=True, check=True)
        print("Image sent to Greenshot editor.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to send image to Greenshot: {e}")

# Track pressed keys for modifier detection
pressed_keys = set()


def on_press(key):
    try:
        pressed_keys.add(key)
        # Shift+Insert: Greenshot flow
        if key == kb.Key.insert and kb.Key.shift in pressed_keys:
            monitor_index = 1
            print(f"(Shift+Insert) Taking screenshot of monitor {monitor_index} and sending to Greenshot...")
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
            send_to_greenshot(image, boxes[0])
            return
        # Insert only: regular flow
        if key == kb.Key.insert and kb.Key.shift not in pressed_keys:
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
            image_with_boxes = draw_boxes(image, boxes)
            cropped_image = crop_and_save_image(image, boxes[0])
            copy_image_to_clipboard(cropped_image)
            return
        if key == kb.Key.esc:
            print("Quitting...")
            return False
    except AttributeError:
        pass

def on_release(key):
    try:
        if key in pressed_keys:
            pressed_keys.remove(key)
    except KeyError:
        pass

def delete_results_folder(folder_path='./Results_for_Detector/'):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' and all its contents have been deleted.")
    else:
        print(f"Folder '{folder_path}' does not exist.")


def main():
    print("Press 'Insert' to take a screenshot and detect objects (regular flow).")
    print("Press 'Shift+Insert' to send cropped region to Greenshot.")
    print("Press 'ESC' to quit.")
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
    delete_results_folder()


if __name__ == "__main__":
    main()
