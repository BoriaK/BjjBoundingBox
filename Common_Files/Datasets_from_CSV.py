import glob
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import ast
import numpy as np
import cv2
import torchvision.transforms.functional as Ftv
import xml.etree.ElementTree as ET

# classes_dict = {"Bed": "0",
#                 "Chair": "1",
#                 "Sofa": "2",
#                 "Swivelchair": "3",
#                 "Table": "4"}


class ObjectDetectDataset(Dataset):
    def __init__(self, args, transforms_=None, mode='train'):
        self.root = args.data
        # self.f_annos = args.annos
        self.transform = transforms_
        self.n_classes = args.n_classes
        self.fnames = sorted(glob.glob(args.data + '/*.jpg'))
        self.f_annos = self.root + '/labels_bjj_combined.csv'
        # self.dtypes = {'float': args.dtype_float,
        #                'long': args.dtype_long,
        #                'short': args.dtype_short}
        self.annos, self.imgs, self.classes, self.areas, self.img_id = self.readAnnos()
        self.mode = mode

    def __getitem__(self, index):
        f_img = self.imgs[index]
        annos = self.annos[index]
        img_cv = cv2.imread(os.path.join(self.root, f_img))[:, :, ::-1]
        classes = self.classes[index]
        area = self.areas[index]
        img_id = self.img_id[index]

        if self.mode == 'train':
            if self.transform:
                bboxes = annos.numpy().copy()
                img, bboxes = self.transform(img_cv.copy(), bboxes.copy())
                img = Ftv.to_tensor(img.copy())
                annos = torch.from_numpy(bboxes.copy())
            else:
                img = Ftv.to_tensor(img_cv.copy())
        else:
            img = Ftv.to_tensor(img_cv.copy())

        # img.type(self.dtypes['float'])
        item = {'boxes': annos,
                'labels': classes,
                'image_id': img_id,
                'area': area,
                'iscrowd': torch.zeros(1)}

        return img, item

    def __len__(self):
        return len(self.fnames)

    def readAnnos(self):
        annos = []
        imgs = []
        classes = []
        areas = []
        f = open(self.f_annos, 'r')
        lines = f.readlines()
        img_id = torch.arange(0, len(lines), 1).type(torch.LongTensor)
        for k, line_ in enumerate(lines):
            if k >= 1:
                line = line_.replace(' ', '')
                img_name = line.split(',')[5]
                annos_k = torch.zeros(1, 4).type(torch.FloatTensor)
                # the anotations come in the following format: x, y, W, H. need to convert to x0,y0,x1,y1
                annos_k[0, :] = torch.Tensor([int(line.split(',')[1]), int(line.split(',')[2]),
                                              int(line.split(',')[1]) + int(line.split(',')[3]),
                                              int(line.split(',')[2]) + int(line.split(',')[4])])
                areas_k = torch.Tensor([int(line.split(',')[3]) * int(line.split(',')[4])])
                classes_k = 1 * torch.ones(1).type(torch.LongTensor)

                annos.append(annos_k)
                imgs.append(img_name)
                classes.append(classes_k)
                areas.append(areas_k)
        return annos, imgs, classes, areas, img_id
