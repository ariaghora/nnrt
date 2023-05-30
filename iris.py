from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

from dump import dump_ndarray_list

x, y = load_iris(return_X_y=True)
y = y.ravel()
x = MinMaxScaler().fit_transform(x)
clf = MLPClassifier(hidden_layer_sizes=(100, 20), max_iter=1000).fit(x, y)

all_params = clf.coefs_ + [i.reshape(1, -1) for i in clf.intercepts_]

dump_ndarray_list(all_params, "mlp.dat")
