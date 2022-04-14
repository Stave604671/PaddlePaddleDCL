import paddle
import paddle.nn as nn
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


class FocalLoss(nn.Layer):  # 1d and 2d

    def __init__(self, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, logit, target, class_weight=None, type='softmax'):
        target = paddle.reshape(target, shape=[-1, 1]).floor()
        #         target = target.view(-1, 1).long()
        if type == 'sigmoid':
            if class_weight is None:
                class_weight = [1] * 2  # [0.5, 0.5]

            prob = paddle.nn.functional.sigmoid(logit)
            prob = paddle.reshape(prob, shape=(-1, 1))
            prob = paddle.concat((1 - prob, prob), 1)
            select = paddle.zeros(shape=[len(prob), 2])

            select.put_along_axis_(indices=target, values=1, axis=1)

        elif type == 'softmax':
            B, C = logit.shape
            if class_weight is None:
                class_weight = [1] * C  # [1/C]*C

            # logit   = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
            prob = F.softmax(logit, 1)
            select = paddle.zeros(shape=[len(prob), C])
            select.put_along_axis_(indices=target, values=1, axis=1)
        class_weight = paddle.to_tensor(class_weight)
        class_weight = paddle.reshape(x=class_weight, shape=[-1, 1])
        # class_weight = paddle.FloatTensor(class_weight).cuda().view(-1, 1)
        class_weight = paddle_gather(x=class_weight, dim=0, index=target)

        prob = (prob * select).sum(1)
        prob = paddle.reshape(x=prob, shape=[-1, 1])
        prob = paddle.clip(prob, 1e-8, 1 - 1e-8)
        batch_loss = - class_weight * (paddle.pow((1 - prob), self.gamma)) * prob.log()

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss

        return loss
