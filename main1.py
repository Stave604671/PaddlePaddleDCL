import paddle
x = paddle.zeros([3, 5], dtype="int64")
updates = paddle.arange(1, 11).reshape([2,5])
# 输出
# Tensor(shape=[2, 5], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
#        [[1 , 2 , 3 , 4 , 5 ],
#         [6 , 7 , 8 , 9 , 10]])
index = paddle.to_tensor([[0, 1, 2], [0, 1, 4]])
i, j = index.shape
grid_x , grid_y = paddle.meshgrid(paddle.arange(i), paddle.arange(j))
# 若 PyTorch 的 dim 取 0
# index = paddle.stack([index.flatten(), grid_y.flatten()], axis=1)
# 若 PyTorch 的 dim 取 1
index = paddle.stack([grid_x.flatten().astype("int64"), index.flatten().astype("int64")], axis=1)
# PaddlePaddle updates 的 shape 大小必须与 index 对应
updates_index = paddle.stack([grid_x.flatten().astype("int64"), grid_y.flatten().astype("int64")], axis=1)
updates = paddle.gather_nd(updates, index=updates_index)
updates_check = paddle.ones(shape=[1])
out = paddle.scatter_nd_add(x=x, index=index, updates=updates_check)
