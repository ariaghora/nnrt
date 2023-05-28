from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from typing import List

import numpy as np


def dump(tensor_list: List[np.ndarray], filename: str) -> None:
    with open(filename, "wb") as f:
        for tensor in tensor_list:
            tensor = np.ascontiguousarray(tensor, dtype=np.float32)
            # ndim
            np.array(tensor.ndim, dtype=np.int32).tofile(f)
            # shape
            np.array(tensor.shape, dtype=np.int32).tofile(f)
            # data
            f.write(tensor.tobytes())


x, y = load_iris(return_X_y=True)
y = y.ravel()
x = MinMaxScaler().fit_transform(x)
clf = MLPClassifier(hidden_layer_sizes=(100, 20), max_iter=1000).fit(x, y)

all_params = clf.coefs_ + [i.reshape(1, -1) for i in clf.intercepts_]

dump(all_params, "mlp.dat")
