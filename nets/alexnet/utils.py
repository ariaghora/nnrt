import torch

from dump.dump import dump_ndarray

model: torch.nn.Module = torch.hub.load(
    "pytorch/vision:v0.10.0", "alexnet", pretrained=True
)
model.eval()
