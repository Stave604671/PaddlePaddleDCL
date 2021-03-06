# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# import pdb
import paddle
from paddle import nn


class LossRecord(object):
    def __init__(self, batch_size):
        self.rec_loss = 0
        self.count = 0
        self.batch_size = batch_size

    def update(self, loss):
        if isinstance(loss, list):
            avg_loss = sum(loss)
            avg_loss /= (len(loss) * self.batch_size)
            self.rec_loss += avg_loss
            self.count += 1
        if isinstance(loss, float):
            self.rec_loss += loss / self.batch_size
            self.count += 1

    def get_val(self, init=False):
        # print("get_val", self.rec_loss, self.count)
        pop_loss = self.rec_loss / self.count
        if init:
            self.rec_loss = 0
            self.count = 0
        return pop_loss


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2D):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = paddle.sqrt(totalnorm).item()
    norm = (clip_norm / max(totalnorm, clip_norm))
    for p in model.parameters():
        if p.requires_grad:
            p.grad.mul_(norm)


def Linear(in_features, out_features, bias=True):
    """Weight-normalized Linear layer (input: N x T x C)"""
    if bias:
        return nn.Linear(in_features, out_features, bias_attr=paddle.nn.initializer.Uniform(-0.1, 0.1),
                         weight_attr=paddle.nn.initializer.Uniform(-0.1, 0.1))
    else:
        return nn.Linear(in_features, out_features, bias_attr=bias,
                         weight_attr=paddle.nn.initializer.Uniform(-0.1, 0.1))


class convolution(nn.Layer):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2D(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride),
                              bias_attr=not with_bn)
        self.bn = nn.BatchNorm2D(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU()

    def forward(self, x):
        conv = self.conv(x)
        bn = self.bn(conv)
        relu = self.relu(bn)
        return relu


class fully_connected(nn.Layer):
    def __init__(self, inp_dim, out_dim, with_bn=True):
        super(fully_connected, self).__init__()
        self.with_bn = with_bn

        self.linear = nn.Linear(inp_dim, out_dim)
        if self.with_bn:
            self.bn = nn.BatchNorm1D(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        linear = self.linear(x)
        bn = self.bn(linear) if self.with_bn else linear
        relu = self.relu(bn)
        return relu


class residual(nn.Layer):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(residual, self).__init__()

        self.conv1 = nn.Conv2D(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride), bias_attr=False)
        self.bn1 = nn.BatchNorm2D(out_dim)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2D(out_dim, out_dim, (3, 3), padding=(1, 1), bias_attr=False)
        self.bn2 = nn.BatchNorm2D(out_dim)

        self.skip = nn.Sequential(
            nn.Conv2D(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias_attr=False),
            nn.BatchNorm2D(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu = nn.ReLU()

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)

        skip = self.skip(x)
        return self.relu(bn2 + skip)
