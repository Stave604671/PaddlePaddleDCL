import paddle
import numpy as np
from paddle.nn import CrossEntropyLoss
ce_loss = CrossEntropyLoss()

input_data = paddle.randn([5, 100], dtype="float32")
label_data = np.random.randint(0, 100, size=(5)).astype(np.int32)
label_data = paddle.to_tensor(label_data)
out = ce_loss(input_data, label_data)