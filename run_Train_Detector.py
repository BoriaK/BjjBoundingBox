import os
import torch
from torch.utils.data import DataLoader
# from Common_Files.Datasets_XML import ObjectDetectDataset
from Common_Files.Datasets_from_CSV import ObjectDetectDataset
import argparse
import torchvision.transforms as transforms
from Common_Files.Models import Detector
import math
import sys
import cv2
import numpy as np
import Common_Files.utils as utils
# from Augmentation.data_aug.data_aug import *
# from Augmentation.data_aug.bbox_util import *
import torch.nn as nn
from torchvision.ops import nms
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--resume_epoch', type=int, default=None, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=4, help='size of the batches')
# parser.add_argument('--data', type=str, default=r'./Dataset/Sub_Set_v01/train/bed', help='root directory of the '
#                                                                                          'dataset')
parser.add_argument('--data', type=str, default=r'./Dataset/Sub_Set_v01/train/bjj/labeled',
                    help='root directory of the dataset')
parser.add_argument('--save_results', action='store_false', default=True,
                    help='save results as images after each training epoch')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--root_chkps', type=str, default='./checkpoints', help='checkpoint folder')
parser.add_argument('--n_classes', type=int, default=2, help='number of total classes')
args = parser.parse_args()
print(args)
'''adding types to arguments'''
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(args.device)

if not os.path.isdir(args.root_chkps):
    os.mkdir(args.root_chkps)

'''train'''
# use our dataset and defined transformations transforms_train = Sequence([RandomHorizontalFlip(0.5), RandomScale(
# 0.5, diff=True), RandomRotate(10), RandomShear(0.2)]) transforms_train = Sequence( [RandomHorizontalFlip(0.5),
# RandomRotate(10), RandomShear(0.2), RandomHSV(30,30,30), RandomColor(0.5), RandomTranslate(0.5)])
transforms_train = None

transforms_test = None


# Debug Path
# root = r'C:\Users\bkupcha\OneDrive - Intel Corporation\Documents\PythonProjects\Furnecher_Project_Pytorch\Dataset' \
#        r'\Sub_Set_v01\train\bed'
# print(os.path.abspath(args.data))


def trainDetector():
    dataset = ObjectDetectDataset(args, transforms_train, 'train')
    dataset_test = ObjectDetectDataset(args, transforms_test, 'test')

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    # indices = list(range(len(dataset)))
    dataset = torch.utils.data.Subset(dataset, indices[:-int(0.1 * len(dataset))])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-int(0.1 * len(dataset_test)):])

    # define training and validation data loaders
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                             collate_fn=utils.collate_fn, drop_last=True)

    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0, collate_fn=utils.collate_fn)

    # get the model using our helper function
    # move model to the right device
    model = Detector(n_classes=args.n_classes, device=args.device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    if args.resume_epoch is not None:
        if not os.path.isfile('./checkpoints/bed_' + str(args.resume_epoch) + '.pth'):
            exit('no pth file')
        checkpoint = torch.load('./checkpoints/bed_' + str(args.resume_epoch) + '.pth')
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state'])
        start_epoch = args.resume_epoch
    else:
        start_epoch = 0

    '''train loop'''
    for epoch in range(start_epoch, args.n_epochs):
        model.train()
        is_train = True
        loss_av = 0.

        for batch_idx, (images, targets) in enumerate(data_loader):
            images = list(image.to(args.device) for image in images)
            #################for Debug################
            # for image in images:
            #     new_image = np.moveaxis(image.numpy(), [0, 1, 2], [2, 0, 1])
            #     plt.imshow(new_image)
            #     plt.show()
            #########################################
            targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]
            # print(targets) # for Debug
            loss_dict = model(images, targets, is_train)
            #################for Debug################
            # print(loss_dict)
            # for loss in loss_dict.values():
            #     print(loss)
            ######################################################
            losses = sum(loss for loss in loss_dict.values())
            # print(losses)   #for Debug

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            loss_av += losses.item()

            if batch_idx % 5 == 0:
                losses2print = [l.item() for l in list(loss_dict.values())]
                print('epoch:{}/{}, batch:{}/{}, lr={:4f}, '
                      'loss-classifier:{:2f}, '
                      'loss_box_reg:{:2f}, '
                      'loss_objectness={:2f}, '
                      'loss_rpn_box_reg={:2f}'.
                      format(epoch, args.n_epochs,
                             batch_idx, len(data_loader),
                             optimizer.param_groups[0]["lr"],
                             losses2print[0],
                             losses2print[1],
                             losses2print[2],
                             losses2print[3]))

        loss_av /= len(data_loader)
        if lr_scheduler is not None:
            lr_scheduler.step()

        # evaluate on the test dataset
        if epoch % 3 == 0 or epoch == args.n_epochs - 1:
            checkpoint = {'epoch': epoch,
                          'model_state': model.state_dict(),
                          'optimizer_state': optimizer.state_dict(),
                          'lr_scheduler_state': lr_scheduler.state_dict()
                          }
            # for Debug dissable saving new check points:
            torch.save(checkpoint, './checkpoints/bjj_' + str(epoch) + '.pth')

        with torch.no_grad():
            model.eval()
            is_train = False
            conf = 0.9
            min_iou = 0.5
            for i, (images, targets), in enumerate(data_loader_test):
                images = list(image.to(args.device) for image in images)
                targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]
                predictions = model(images, [], is_train)
                boxes = predictions[0]['boxes'].cpu().numpy() if predictions[0]['boxes'].is_cuda else predictions[0][
                    'boxes'].numpy()
                labels = predictions[0]['labels'].cpu().numpy() if predictions[0]['labels'].is_cuda else predictions[0][
                    'labels'].numpy()
                scores = predictions[0]['scores'].cpu().numpy() if predictions[0]['scores'].is_cuda else predictions[0][
                    'scores'].numpy()
                boxes_keep = nms(torch.from_numpy(boxes), torch.from_numpy(scores), 0.5)
                boxes = boxes[boxes_keep.numpy(), :]
                labels = labels[boxes_keep.numpy()]
                scores = scores[boxes_keep.numpy()]
                im_ = images[0].cpu().numpy() if images[0].is_cuda else images[0].numpy()
                im_ = np.moveaxis(im_, 0, -1)[:, :, ::-1]
                im_ = np.ascontiguousarray(im_)  # this is due to a memorry related assert in the code
                im_ = (im_ * 255).astype(np.uint8)
                for j, b in enumerate(boxes):
                    if scores[j] > conf:
                        print('detected')
                        im_ = cv2.rectangle(im_, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 5)
                        im_ = cv2.putText(im_, str(np.round(scores[j] * 100) / 100), (int(b[0]), int(b[3])),
                                          cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (255, 0, 0), 10, cv2.LINE_AA)
                # for Debug disable saving the image with bounding box, and just display it:
                ########################################################################################
                if args.save_results:
                    if not os.path.isdir(r'./Results_for_Detector_and_Classifier/'):
                        os.mkdir('./Results_for_Detector_and_Classifier/')
                    cv2.imwrite(
                        './Results_for_Detector_and_Classifier/' + 'epoch' + str(epoch) + '_Img' + str(i) + '.jpg',
                        im_)
                ###########################################################
                # cv2.namedWindow('../Results/' + str(i),
                #                 cv2.WINDOW_NORMAL)  # to make the image fit the display window when printing
                # cv2.imshow('../Results/' + str(i), im_)
                # plt.show()


if __name__ == "__main__":
    trainDetector()
