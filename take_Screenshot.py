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
from PIL import Image, ImageOps, ImageGrab
import glob
import xml.etree.ElementTree as ET
from Common_Files.ElementTree_pretty import prettify
import csv
import screeninfo
import pyautogui
import keyboard
import pygetwindow as gw
import mss

# from yattag import Doc, indent

# This script is testing the detector, to detect Furniture and place bounding box around them


# Get all monitors information
with mss.mss() as sct:
    # Get all monitors
    monitors = sct.monitors

    for i, monitor in enumerate(monitors):
        print(f"Monitor: {i}, Left: {monitor['left']}, Top: {monitor['top']}, Width: {monitor['width']}, Height: {monitor['height']}")

    # Get the monitor at the specified index
    index = 1
    monitor = monitors[index]
    # a correction to the partial screenshot on the small monitor
    if len(monitors) > 2 and index == 1:
        monitor['width'] = monitor['width'] + 380
        monitor['height'] = monitor['height'] + 220

    # Take a screenshot
    screenshot = sct.grab(monitor)
    # Convert the screenshot to a PIL image
    screenshot_image = Image.frombytes("RGB", screenshot.size, screenshot.rgb)

    # for debug:
    screenshot_image.show(title="Screenshot")
    ##################################

    # to check change of colors issue
    im_npArray = np.array(screenshot_image)
    # im = Ftv.to_tensor(screenshot_image)

    # Convert BGR back to RGB
    im_npArray = cv2.cvtColor(im_npArray, cv2.COLOR_BGR2RGB)

    im = cv2.rectangle(im_npArray, (100, 500),
                       (500, 100), (255, 0, 0), 5)
    im = cv2.putText(im_npArray, str(1), (100, 800),
                     cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (255, 0, 0), 10, cv2.LINE_AA)

    # cv2.imwrite('./Results_for_Detector_and_Classifier/' + 'from_screenShot.png', im)

    # for visualization:
    cv2.namedWindow('from_screenShot.png', cv2.WINDOW_NORMAL)  # to make the image fit the display window when printing
    cv2.imshow('from_screenShot.png', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    plt.show()
