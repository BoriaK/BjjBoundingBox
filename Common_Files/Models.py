import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class Detector(nn.Module):
    # def __init__(self, n_classes, device, dtype_float):
    def __init__(self, n_classes, device):
        super(Detector, self).__init__()
        # load a model pre-trained pre-trained on COCO
        # self.det_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True) # old version
        self.det_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.n_classes = n_classes
        self.device = device
        # replace the classifier with a new one, that has
        # num_classes which is user-defined
        num_classes = 2  # 1 class (person) + background
        # get number of input features for the classifier
        in_features = self.det_model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.det_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)
        self.det_model.to(device)

    def forward(self, x, t, is_train):
        if is_train:
            pred = self.det_model(x, t)
        else:
            pred = self.det_model(x)
        return pred


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding=0, dilation=1):
        super(ConvBlock, self).__init__()

        conv_block = [
            nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
            nn.InstanceNorm2d(c_out),
            nn.ELU(inplace=True)]
        self.conv_block = nn.Sequential(*conv_block)
