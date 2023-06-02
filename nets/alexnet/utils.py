import os

import torch
import urllib

from dump.dump import dump_ndarray, dump_torch_conv2d, dump_torch_linear

from PIL import Image
from torchvision import transforms


model_dir = os.path.dirname(__file__)
model_path = os.path.join(model_dir, "alexnet.dat")

"""===================================================================================
AlexNet(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    (1): relu(inplace=true)
    (2): maxpool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=false)

    (3): conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (4): relu(inplace=true)
    (5): maxpool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=false)

    (6): conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): relu(inplace=true)

    (8): conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): relu(inplace=true)

    (10): conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): relu(inplace=true)
    (12): maxpool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=false)
  )
  (avgpool): adaptiveavgpool2d(output_size=(6, 6))
  (classifier): sequential(
    (0): dropout(p=0.5, inplace=false)
    (1): linear(in_features=9216, out_features=4096, bias=true)
    (2): relu(inplace=true)
    (3): dropout(p=0.5, inplace=false)
    (4): linear(in_features=4096, out_features=4096, bias=true)
    (5): relu(inplace=true)
    (6): linear(in_features=4096, out_features=1000, bias=true)
  )
)
======================================================================================"""

model: torch.nn.Module = torch.hub.load(
    "pytorch/vision:v0.10.0", "alexnet", pretrained=True
)
model.eval()

# We only care about saving convolution and linear layer weights.
# In evaluation phase, dropout will be ignored. The rest type of layers
# are without parameters and nnrt has the functionalities already.

convs = [
    model.features[0],
    model.features[3],
    model.features[6],
    model.features[8],
    model.features[10],
]

linears = [
    model.classifier[1],
    model.classifier[4],
    model.classifier[6],
]

with open(model_path, "wb") as f:
    for c in convs:
        dump_torch_conv2d(c, f)
    for l in linears:
        dump_torch_linear(l, f)
