from io import BufferedWriter
from typing import List

import numpy as np
from torch.nn import Conv2d, Linear


def dump_i32(x: int, f: BufferedWriter):
    np.array(x, dtype=np.int32).tofile(f)


def dump_ndarray(x: np.ndarray, f: BufferedWriter):
    x = np.ascontiguousarray(x, dtype=np.float32)
    np.array(x.ndim, dtype=np.int32).tofile(f)
    np.array(x.shape, dtype=np.int32).tofile(f)
    f.write(x.tobytes())


def dump_torch_conv2d(layer: Conv2d, f: BufferedWriter):
    w = layer.weight.detach().numpy()
    b = layer.bias.detach().numpy()
    stride = layer.stride

    sx, sy = layer.stride
    if sx != sy:
        raise ValueError("Asymmetric stride is not supported yet")
    stride = sx

    px, py = layer.padding
    if px != py:
        raise ValueError("Asymmetric padding is not supported yet")
    pad = px

    dump_ndarray(w, f)
    dump_ndarray(b, f)
    dump_i32(stride, f)
    dump_i32(pad, f)


def dump_torch_linear(layer: Linear, f: BufferedWriter):
    w = layer.weight.detach().numpy()
    b = layer.bias.detach().numpy()
    dump_ndarray(w, f)
    dump_ndarray(b, f)


def dump_ndarray_list(ndarray_list: List[np.ndarray], filename: str) -> None:
    with open(filename, "wb") as f:
        for x in ndarray_list:
            dump_ndarray(x, f)
