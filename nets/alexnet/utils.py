import torch

from dump.dump import dump_ndarray, dump_torch_conv2d, dump_torch_linear

model: torch.nn.Module = torch.hub.load(
    "pytorch/vision:v0.10.0", "alexnet", pretrained=True
)
model.eval()

import os

model_path = os.path.join(os.path.dirname(__file__), "alexnet.dat")

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
