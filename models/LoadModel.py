import numpy as np
# from torch import nn
# import torch
# from torchvision import models, transforms, datasets
# import torch.nn.functional as F
# import pretrainedmodels
import paddle
from paddle.vision import models, transforms, datasets
from config import pretrained_model
import paddle.nn.functional as F
from paddle import nn
import pdb
from models.Asoftmax_linear import AngleLinear

class MainModel(nn.Layer):
    def __init__(self, config):
        super(MainModel, self).__init__()
        self.use_dcl = config.use_dcl
        self.num_classes = config.numcls
        self.backbone_arch = config.backbone
        self.use_Asoftmax = config.use_Asoftmax
        print(self.backbone_arch)

        if self.backbone_arch in dir(models):
            self.model = getattr(models, self.backbone_arch)()
            if self.backbone_arch in pretrained_model:
                self.model.set_state_dict(paddle.load(r"DCL-master/pretrain/resnet50.pdparams"))

        if self.backbone_arch == 'resnet50' or self.backbone_arch == 'se_resnet50':
            self.model = nn.Sequential(*list(self.model.children())[:-2])
        if self.backbone_arch == 'senet154':
            self.model = nn.Sequential(*list(self.model.children())[:-3])
        if self.backbone_arch == 'se_resnext101_32x4d':
            self.model = nn.Sequential(*list(self.model.children())[:-2])
        if self.backbone_arch == 'se_resnet101':
            self.model = nn.Sequential(*list(self.model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2D(output_size=1)
        self.classifier = nn.Linear(2048, self.num_classes, bias_attr=False)

        if self.use_dcl:
            if config.cls_2:
                self.classifier_swap = nn.Linear(2048, 2, bias_attr=False)
            if config.cls_2xmul:
                self.classifier_swap = nn.Linear(2048, 2*self.num_classes, bias_attr=False)
            self.Convmask = nn.Conv2D(2048, 1, 1, stride=1, padding=0, bias_attr=True)
            self.avgpool2 = nn.AvgPool2D(2, stride=2)

        if self.use_Asoftmax:
            self.Aclassifier = AngleLinear(2048, self.num_classes)  # , bias_=False)

    def forward(self, x, last_cont=None):
        x = self.model(x)
        if self.use_dcl:
            mask = self.Convmask(x)
            mask = self.avgpool2(mask)
            mask = paddle.tanh(mask)
            mask = paddle.reshape(x=mask, shape=[mask.shape[0], -1])

        x = self.avgpool(x)
        x = paddle.reshape(x, shape=[x.shape[0], -1])
        out = []
        out.append(self.classifier(x))

        if self.use_dcl:
            out.append(self.classifier_swap(x))
            out.append(mask)

        if self.use_Asoftmax:
            if last_cont is None:
                x_size = x.size(0)
                out.append(self.Aclassifier(x[0:x_size:2]))
            else:
                last_x = self.model(last_cont)
                last_x = self.avgpool(last_x)
                last_x = last_x.view(last_x.size(0), -1)
                out.append(self.Aclassifier(last_x))

        return out
