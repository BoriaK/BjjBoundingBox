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
root = r'./Dataset/Sub_Set_v01'

# Initialize the CSV file and write the header
with open('Results_for_Detector/detection_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["label_name", "bbox_x", "bbox_y", "bbox_width", "bbox_height", "image_name", "score"])

    for img in glob.glob(os.path.join(root, r'test/*.jpg')):
        # im_cv = cv2.imread(os.path.join(root, 'for_Test_Detector/DSCF1025.JPG'))
        im_cv = cv2.imread(img)
        # im_cv = cv2.resize(im_cv, (2736, 3648), interpolation=cv2.INTER_LINEAR)
        fimg = Image.fromarray(im_cv)
        # im = Ftv.to_tensor(fimg).type(dtype_float)
        im = Ftv.to_tensor(fimg)
        # im = Ftv.to_tensor(im_cv)
        predictions = model([im.to(device)], [], is_train)
        boxes = predictions[0]['boxes'].detach().cpu().numpy() if predictions[0]['boxes'].is_cuda else predictions[0][
            'boxes'].detach().numpy()
        labels = predictions[0]['labels'].detach().cpu().numpy() if predictions[0]['labels'].is_cuda else predictions[0][
            'labels'].detach().numpy()
        scores = predictions[0]['scores'].detach().cpu().numpy() if predictions[0]['scores'].is_cuda else predictions[0][
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
        # im_ = im[0].detach().cpu().numpy() if im[0].is_cuda else im[0].detach().numpy()
        # print(im_.shape)
        # im_ = np.moveaxis(im_, 0, -1)[:, :, ::-1]
        # im_ = np.ascontiguousarray(im_)  # this is due to a memory related assert in the code
        # im_ = (im_ * 255).astype(np.uint8)
        conf = 0.97

        for j in range(boxes.shape[0]):
            b = boxes[j, :].copy()
            if scores[j] > conf:
                print('detected')
                print(b)
                b_xml = b
                im = cv2.rectangle(im_cv, (int(np.round(b[0])), int(np.round(b[1]))),
                                   (int(np.round(b[2])), int(np.round(b[3]))), (0, 255, 0), 5)
                im = cv2.putText(im_cv, str(np.round(scores[j] * 100) / 100), (int(b[0]), int(b[3])),
                                 cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (255, 0, 0), 10, cv2.LINE_AA)
        elapsed = time.time() - ts
        print('elapsed={}'.format(elapsed))
        # cv2.imwrite('./Results_for_Detector/' + 'testing' + img.split("\\")[-1], im_cv)
        cv2.imwrite('./Results_for_Detector/' + 'testing' + img.split("\\")[-1], im)
        # for visualization:
        # cv2.namedWindow(img.split("\\")[-1], cv2.WINDOW_NORMAL)  # to make the image fit the display window when printing
        # # cv2.imshow(img.split("\\")[-1], im_cv)
        # cv2.imshow(img.split("\\")[-1], im)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # plt.show()

        # create .csv output with estimated BBox
        writer.writerow(['bjj', np.round(b[0]), np.round(b[1]), im_cv.shape[1], im_cv.shape[0], img.split('\\')[-1], scores[0]])
        print(img.split('\\')[-1])
        print('')

        # create .xml output with estimated BBox
        # img_xml = os.path.abspath(img)
        # print(img_xml)

        # option 1 - yattag:

        # doc, tag, text = Doc().tagtext()
        #
        # with tag('annotation'):
        #     with tag('folder'):
        #         text(img_xml.split("\\")[-2])
        #     with tag('filename'):
        #         text(img_xml.split("\\")[-1])
        #     with tag('path'):
        #         text(img_xml)
        #     with tag('source'):
        #         with tag('database'):
        #             text('Unknown')
        #     with tag('size'):
        #         with tag('width'):
        #             text(str(im_cv.shape[1]))
        #         with tag('height'):
        #             text(str(im_cv.shape[0]))
        #         with tag('depth'):
        #             text(str(im_cv.shape[2]))
        #     with tag('segmented'):
        #         text('0')
        #     with tag('object'):
        #         with tag('name'):
        #             text('bed')
        #         with tag('pose'):
        #             text('Unspecified')
        #         with tag('truncated'):
        #             text('0')
        #         with tag('difficult'):
        #             text('0')
        #         with tag('bndbox'):
        #             with tag('xmin'):
        #                 text(str(int(np.round(b_xml[0]))))
        #             with tag('ymin'):
        #                 text(str(int(np.round(b_xml[1]))))
        #             with tag('xmax'):
        #                 text(str(int(np.round(b_xml[2]))))
        #             with tag('ymax'):
        #                 text(str(int(np.round(b_xml[3]))))
        #
        # output_xml = indent(
        #     doc.getvalue(),
        #     indentation=' ' * 4,
        #     # newline='\r\n'  # to leave a single line space between lines...
        #     newline='\n'
        # )
        #
        # print(output_xml)

        # # option 2 ET:
        # top = ET.Element('annotation')
        #
        # folder = ET.SubElement(top, 'folder')
        # if device == "cpu":
        #     folder.text = img_xml.split("\\")[-2]
        # else:
        #     folder.text = img_xml.split("/")[-2]  # for collab
        # filename = ET.SubElement(top, 'filename')
        # if device == "cpu":
        #     filename.text = img_xml.split("\\")[-1]
        # else:
        #     filename.text = img_xml.split("/")[-1]  # for collab
        # path = ET.SubElement(top, 'path')
        # path.text = img_xml
        #
        # source = ET.SubElement(top, 'source')
        # database = ET.SubElement(source, 'database')
        # database.text = 'Unknown'
        #
        # size = ET.SubElement(top, 'size')
        # width = ET.SubElement(size, 'width')
        # width.text = str(im_cv.shape[1])
        # height = ET.SubElement(size, 'height')
        # height.text = str(im_cv.shape[0])
        # depth = ET.SubElement(size, 'depth')
        # depth.text = str(im_cv.shape[2])
        #
        # segmented = ET.SubElement(top, 'segmented')
        # segmented.text = '0'
        #
        # object = ET.SubElement(top, 'object')
        # name = ET.SubElement(object, 'name')
        # name.text = 'bed'
        # pose = ET.SubElement(object, 'pose')
        # pose.text = 'Unspecified'
        # truncated = ET.SubElement(object, 'truncated')
        # truncated.text = '0'
        # difficult = ET.SubElement(object, 'difficult')
        # difficult.text = '0'
        # bndbox = ET.SubElement(object, 'bndbox')
        # xmin = ET.SubElement(bndbox, 'xmin')
        # xmin.text = str(int(np.round(b_xml[0])))
        # ymin = ET.SubElement(bndbox, 'ymin')
        # ymin.text = str(int(np.round(b_xml[1])))
        # xmax = ET.SubElement(bndbox, 'xmax')
        # xmax.text = str(int(np.round(b_xml[2])))
        # ymax = ET.SubElement(bndbox, 'ymax')
        # ymax.text = str(int(np.round(b_xml[3])))

        # output_xml = prettify(top)

        # print(output_xml)

        # new_xml = open(img_xml.split(".")[0] + '.xml', 'w')
        # new_xml.write(output_xml)
        # new_xml.close()
