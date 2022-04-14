import math
import paddle
from paddle import nn
import paddle.nn.functional as F


def myphi(x,m):
    x = x * m
    return 1-x**2/math.factorial(2)+x**4/math.factorial(4)-x**6/math.factorial(6) + \
            x**8/math.factorial(8) - x**9/math.factorial(9)


class AngleLinear(nn.Layer):
    def __init__(self, in_features, out_features, m = 4, phiflag=True):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # self.weight = paddle.zeros(shape=)
        data_pd = paddle.nn.functional.normalize(paddle.uniform(shape=[in_features, out_features], min=-1, max=1), p=2, axis=1, epsilon=1e-5)
        data_pd_mul = paddle.multiply(data_pd, paddle.to_tensor([1e5]))
        self.weight = paddle.create_parameter(shape=[in_features, out_features], dtype="float16",
                                              default_initializer=paddle.nn.initializer.Assign(value=data_pd_mul))
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]

    def forward(self, input):
        x = input   # size=(B,F)    F is feature len
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features
        w = F.normalize(w, p=2, axis=1, epsilon=1e-5)
        ww = paddle.multiply(w, paddle.to_tensor([1e5]))
        xlen = x.pow(2).sum(1).pow(0.5)  # size=B
        wlen = ww.pow(2).sum(0).pow(0.5)  # size=Classnum
        cos_theta = x.mm(ww)  # size=(B,Classnum)
        cos_theta = cos_theta / paddle.reshape(xlen, shape=[-1, 1]) / paddle.reshape(wlen, shape=[1, -1])
        cos_theta = cos_theta.clip(-1, 1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = paddle.acos(cos_theta)
            # theta = Variable(cos_theta.data.acos())
            k = (self.m*theta/3.14159265).floor()
            n_one = k*0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2*k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta,self.m)
            phi_theta = phi_theta.clip(-1*self.m, 1)

        cos_theta = cos_theta * paddle.reshape(xlen, shape=(-1,1))
        phi_theta = phi_theta * paddle.reshape(xlen, shape=(-1,1))
        output = (cos_theta,phi_theta)
        return output # size=(B,Classnum,2)


