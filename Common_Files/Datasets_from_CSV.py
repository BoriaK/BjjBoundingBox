import csv
import glob
import os
import torch
from torchvision import tv_tensors
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
        img_path = os.path.join(self.root, f_img)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError('cannot read image: ' + img_path)
        img_cv = np.ascontiguousarray(img_bgr[:, :, ::-1])
        classes = self.classes[index]
        area = self.areas[index]
        img_id = self.img_id[index]

        img = None
        if self.mode == 'train' and self.transform:
            # v2 transforms only touch tv_tensors; plain numpy inputs pass through unchanged
            img_tv = tv_tensors.Image(torch.from_numpy(img_cv).permute(2, 0, 1))
            boxes_tv = tv_tensors.BoundingBoxes(annos.clone(), format='XYXY', canvas_size=img_tv.shape[-2:])
            img_aug, boxes_aug = self.transform(img_tv, boxes_tv)
            boxes_aug = boxes_aug.as_subclass(torch.Tensor)
            wh = boxes_aug[:, 2:] - boxes_aug[:, :2]
            # keep the augmentation only if the box survived it (e.g. wasn't rotated out of the frame)
            if bool((wh >= 1).all()):
                img = img_aug.as_subclass(torch.Tensor).float().div(255)
                annos = boxes_aug
                area = (wh[:, 0] * wh[:, 1]).float()
        if img is None:
            img = Ftv.to_tensor(img_cv)

        # img.type(self.dtypes['float'])
        item = {'boxes': annos,
                'labels': classes,
                'image_id': img_id,
                'area': area,
                'iscrowd': torch.zeros(1)}

        return img, item

    def __len__(self):
        return len(self.imgs)

    def readAnnos(self):
        annos = []
        imgs = []
        classes = []
        areas = []
        required = ('bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'image_name')
        with open(self.f_annos, 'r', newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [name.strip() for name in (reader.fieldnames or [])]
            missing = [name for name in required if name not in reader.fieldnames]
            if missing:
                raise ValueError('{} is missing columns: {}'.format(self.f_annos, ', '.join(missing)))
            for row in reader:
                row = {k: (v or '').strip() for k, v in row.items() if k is not None}
                if not any(row.values()):
                    continue
                try:
                    x, y, w, h = (int(float(row[name])) for name in required[:4])
                except ValueError:
                    raise ValueError('{}: invalid bounding box on line {}'.format(self.f_annos, reader.line_num))
                img_name = row['image_name']
                if w <= 0 or h <= 0:
                    print('Warning: skipping {} (empty bounding box)'.format(img_name))
                    continue
                if not os.path.isfile(os.path.join(self.root, img_name)):
                    print('Warning: skipping {} (image file not found)'.format(img_name))
                    continue
                annos_k = torch.zeros(1, 4).type(torch.FloatTensor)
                # the anotations come in the following format: x, y, W, H. need to convert to x0,y0,x1,y1
                annos_k[0, :] = torch.Tensor([x, y, x + w, y + h])
                areas_k = torch.Tensor([w * h])
                classes_k = 1 * torch.ones(1).type(torch.LongTensor)

                annos.append(annos_k)
                imgs.append(img_name)
                classes.append(classes_k)
                areas.append(areas_k)
        img_id = torch.arange(0, len(imgs), 1).type(torch.LongTensor)
        return annos, imgs, classes, areas, img_id
