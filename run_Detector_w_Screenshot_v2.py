import os
import torch
import torchvision
from torch.utils.data import DataLoader
from Common_Files.Datasets_from_XML import ObjectDetectDataset
from Common_Files.Models import Detector
import sys
import cv2
import numpy as np
import torchvision.transforms.functional as Ftv
from torchvision.ops import nms
import time
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import glob
import xml.etree.ElementTree as ET
from Common_Files.ElementTree_pretty import prettify
import csv
import screeninfo
import pyautogui
import pygetwindow as gw
import mss
from pynput import keyboard
import win32clipboard
import win32con
import io

# from yattag import Doc, indent

# This script is testing the detector, to detect Furniture and place bounding box around them

'''adding types to arguments'''
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

n_classes = 2
is_train = False
model = Detector(n_classes, device)
if device == "cpu":
    checkpoint = torch.load('./checkpoints/bjj_9' + '.pth')
else:
    checkpoint = torch.load('./checkpoints/bjj_9' + '.pth')
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
    # Save the cropped image
    cropped_image.save('./Results_for_Detector_and_Classifier/' + filename)
    print(f"Cropped image saved as {filename}")

    return cropped_image

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

def on_press(key):
    try:
        if key.char == 'u':
            monitor_index = 1  # Change this to the index of the screen you want to capture
            print(f"Taking screenshot of monitor {monitor_index}...")
            image = take_screenshot(monitor_index)
            print("Detecting objects...")
            predictions = detect_objects(image)
            boxes, labels, scores = process_predictions(predictions)
            if not boxes:
                print("No objects detected. Quitting...")
                return False  # Stop the listener and quit the script

            print(f"Detected {len(boxes)} objects:")
            for i in range(len(boxes)):
                print(f"Object {i + 1}:")
                print(f"  Bounding Box: {boxes[i]}")
                print(f"  Label: {labels[i]}")
                print(f"  Score: {scores[i]:.4f}")

            # Draw boxes on the image
            image_with_boxes = draw_boxes(image, boxes)

            # Display the image with bounding boxes using OpenCV
            # cv2.namedWindow('Screenshot with Bounding Boxes',
            #                 cv2.WINDOW_NORMAL)  # to make the image fit the display window when printing
            # cv2.imshow("Screenshot with Bounding Boxes", image_with_boxes)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # plt.show()

            # cv2.imwrite('./Results_for_Detector_and_Classifier/' + 'from_screenShot.png', image_with_boxes)

            # Crop and save the image using the bounding box of the highest scored object
            cropped_image = crop_and_save_image(image, boxes[0])
            copy_image_to_clipboard(cropped_image)

        # elif key.char == 'q':
        if key.char == 'q':
            print("Quitting...")
            return False

    except AttributeError:
        pass

def main():
    print("Press 'u' to take a screenshot and detect objects.")
    print("Press 'q' to quit.")

    # screens = get_screen_info()
    # print("Available monitors:")u
    # for i, screen in enumerate(screens):
    #     print(f"Monitor {i}: Resolution: {screen['width']}x{screen['height']}")

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

if __name__ == "__main__":
    main()

