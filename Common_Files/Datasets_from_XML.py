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

classes_dict = {"Bed": "0",
                "Chair": "1",
                "Sofa": "2",
                "Swivelchair": "3",
                "Table": "4"}


class ObjectDetectDataset(Dataset):
    def __init__(self, args, transforms_=None, mode='train'):
        self.root = args.data
        self.transform = transforms_
        self.n_classes = args.n_classes
        self.fnames = sorted(glob.glob(self.root + '/*.jpg'))
        self.fannos = sorted(glob.glob(self.root + '/*.xml'))
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
        img_id = torch.arange(0, len(self.fannos), 1).type(torch.LongTensor)
        for fanno in self.fannos:
            anno = ET.parse(fanno).getroot()
            img_name = anno[1].text  # to get the image name with sufix
            annos_k2 = torch.zeros(1, 4).type(torch.FloatTensor)
            annos_k2[0, :] = torch.Tensor(
                [int(anno[6][4][0].text), int(anno[6][4][1].text), int(anno[6][4][2].text), int(anno[6][4][3].text)])
            areas_k = torch.Tensor([int(anno[4][0].text) * int(anno[4][1].text)])
            classes_k = 1 * torch.ones(1).type(torch.LongTensor)
            # classes_k = 1
            # for i, p in enumerate(annos_k):
            #     pp = [p[0], p[1], p[0] + p[2], p[1] + p[3]]
            #     annos_k2[i, :] = torch.Tensor(pp)
            #     areas_k[i] = torch.Tensor([p[2] * p[3]])
            #     if self.n_classes == 6:
            #         classes_k[i] = p[-1] - 1
            #     elif self.n_classes == 2:
            #         classes_k[i] = 1
            #     else:
            #         exit('invalid n_classes')
            annos.append(annos_k2)
            imgs.append(img_name)
            classes.append(classes_k)
            areas.append(areas_k)
        return annos, imgs, classes, areas, img_id
