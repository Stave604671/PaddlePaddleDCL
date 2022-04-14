# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# import torch.nn.functional as F
# from torch.nn import Parameter
import paddle
from paddle import nn
import paddle.nn.functional as F


def paddle_gather(x, dim, index):
    index_shape = index.shape
    index_flatten = index.flatten()
    if dim < 0:
        dim = len(x.shape) + dim
    nd_index = []
    for k in range(len(x.shape)):
        if k == dim:
            nd_index.append(index_flatten)
        else:
            reshape_shape = [1] * len(x.shape)
            reshape_shape[k] = x.shape[k]
            x_arange = paddle.arange(x.shape[k], dtype=index.dtype)
            x_arange = x_arange.reshape(reshape_shape)
            dim_index = paddle.expand(x_arange, index_shape).flatten()
            nd_index.append(dim_index)
    ind2 = paddle.transpose(paddle.stack(nd_index), [1, 0]).astype("int64")
    paddle_out = paddle.gather_nd(x, ind2).reshape(index_shape)
    return paddle_out


class AngleLoss(nn.Layer):
    def __init__(self, gamma=0):
        super(AngleLoss, self).__init__()
        self.gamma = gamma
        self.it = 0
        self.LambdaMin = 50.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target, decay=None):
        self.it += 1
        cos_theta, phi_theta = input
        target = paddle.reshape(target, shape=[-1, 1])  # size=(B,1)

        index = cos_theta.data * 0.0  # size=(B,Classnum)
        index.put_along_axis_(indices=paddle.reshape(target, shape=[-1, 1]), values=1, axis=1)

        if decay is None:
            self.lamb = max(self.LambdaMin, self.LambdaMax / (1 + 0.1 * self.it))
        else:
            self.LambdaMax *= decay
            self.lamb = max(self.LambdaMin, self.LambdaMax)
        output = cos_theta * 1.0  # size=(B,Classnum)
        output[index] -= cos_theta[index] * (1.0 + 0) / (1 + self.lamb)
        output[index] += phi_theta[index] * (1.0 + 0) / (1 + self.lamb)

        logpt = F.log_softmax(output, 1)
        logpt = paddle_gather(logpt, dim=1, index=target)
        logpt = paddle.reshape(x=logpt, shape=[-1])
        pt = logpt.exp()

        loss = -1 * (1 - pt) ** self.gamma * logpt
        loss = loss.mean()

        return loss
