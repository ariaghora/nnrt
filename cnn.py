import torch
from torch.nn import Conv2d


c1 = Conv2d(1, 32, kernel_size=3, stride=1, padding=0)
c2 = Conv2d(32, 16, kernel_size=3, stride=1, padding=0)

x = torch.randn((1, 28, 28))
x = c1(x)

print(x.shape)
