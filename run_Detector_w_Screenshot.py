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
import keyboard
import pygetwindow as gw
import mss

# from yattag import Doc, indent

# This script is testing the detector, to detect Furniture and place bounding box around them

'''adding types to arguments'''
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# dtype_float = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
# dtype_long = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
# dtype_short = torch.cuda.ShortTensor if torch.cuda.is_available() else torch.ShortTensor

n_classes = 2
is_train = False
model = Detector(n_classes, device)
if device == "cpu":
    checkpoint = torch.load('./checkpoints/bjj_9' + '.pth')
else:
    checkpoint = torch.load('./checkpoints/bjj_9' + '.pth')
model.load_state_dict(checkpoint['model_state'])
model.eval()
ts = time.time()
# root = r'./Dataset/Sub_Set_v01'

# Get all monitors information
with mss.mss() as sct:
    # Get all monitors
    monitors = sct.monitors

    for i, monitor in enumerate(monitors):
        print(
            f"Monitor: {i}, Left: {monitor['left']}, Top: {monitor['top']}, Width: {monitor['width']}, Height: {monitor['height']}")

    # Get the monitor at the specified index
    index = 1  # index 0 always includes both monitors
    monitor = monitors[index]
    # a correction to the partial screenshot on the small monitor
    if len(monitors) > 2 and index == 1:
        monitor['width'] = monitor['width'] + 380
        monitor['height'] = monitor['height'] + 220

    print("Press 'u' to take a screenshot and detect objects.")
    while True:
        if keyboard.is_pressed('u'):
            print("Taking screenshot...")

            # Take a screenshot
            screenshot = sct.grab(monitor)
            # Convert the screenshot to a PIL image
            screenshot_PIL = Image.frombytes("RGB", screenshot.size, screenshot.rgb)

            # for debug:
            # screenshot_PIL.show(title="Screenshot")
            ##################################

            im_detect = Ftv.to_tensor(screenshot_PIL)
            predictions = model([im_detect.to(device)], [], is_train)
            boxes = predictions[0]['boxes'].detach().cpu().numpy() if predictions[0]['boxes'].is_cuda else \
            predictions[0][
                'boxes'].detach().numpy()
            labels = predictions[0]['labels'].detach().cpu().numpy() if predictions[0]['labels'].is_cuda else \
            predictions[0][
                'labels'].detach().numpy()
            scores = predictions[0]['scores'].detach().cpu().numpy() if predictions[0]['scores'].is_cuda else \
            predictions[0][
                'scores'].detach().numpy()
            boxes_keep = nms(torch.from_numpy(boxes), torch.from_numpy(scores), 0.99)
            boxes = boxes[boxes_keep.numpy(), :]
            labels = labels[boxes_keep.numpy()]
            scores = scores[boxes_keep.numpy()]
            # print(i, boxes_keep, scores.shape, boxes.shape, labels.shape)
            #############################################
            if len(boxes.shape) == 1:
                boxes = boxes.reshape(-1, boxes.shape[0])
            else:
                boxes = boxes.reshape(-1, boxes.shape[1])
            ##############################################
            conf = 0.97
            for j in range(boxes.shape[0]):
                b = boxes[j, :].copy()
                if scores[j] > conf:
                    print('detected')
                    print(b)
                    b_xml = b
                    im_npArray = np.array(screenshot_PIL)
                    # Convert BGR back to RGB
                    im_npArray = cv2.cvtColor(im_npArray, cv2.COLOR_BGR2RGB)
                    im = cv2.rectangle(im_npArray, (int(np.round(b[0])), int(np.round(b[1]))),
                                       (int(np.round(b[2])), int(np.round(b[3]))), (0, 255, 0), 5)
                    # im = cv2.putText(im_npArray, str(np.round(scores[j] * 100) / 100), (int(b[0]), int(b[3])),
                    #                  cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (255, 0, 0), 10, cv2.LINE_AA)

            elapsed = time.time() - ts
            print('elapsed={}'.format(elapsed))
            # cv2.imwrite('./Results_for_Detector_and_Classifier/' + 'from_screenShot.png', im)
            # for visualization:
            cv2.namedWindow('from_screenShot.png',
                            cv2.WINDOW_NORMAL)  # to make the image fit the display window when printing
            cv2.imshow('from_screenShot.png', im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            plt.show()

            # # create .csv output with estimated BBox
            # writer.writerow(['bjj', np.round(b[0]), np.round(b[1]), im_cv.shape[1], im_cv.shape[0], img.split('\\')[-1], scores[0]])
            # print(img.split('\\')[-1])
            # print('')

            print("Press 'u' to take a screenshot and detect objects.")
            print("Press 'q' to stop the program.")

        if keyboard.is_pressed('q'):
            print("Quitting...")
            break
