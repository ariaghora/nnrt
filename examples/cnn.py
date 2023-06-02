import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.nn import Conv2d, CrossEntropyLoss, Linear, Module
from torch.optim import Adam

from dump.dump import dump_ndarray, dump_torch_conv2d, dump_torch_linear

WEIGHT_PATH = "cnn.dat"
X_TEST_PATH = "x_test.dat"
Y_TEST_PATH = "y_test.dat"


class Classifier(Module):
    def __init__(self) -> None:
        super().__init__()
        self.c1 = Conv2d(1, 16, kernel_size=3, stride=1, padding=0)
        self.c2 = Conv2d(16, 8, kernel_size=3, stride=1, padding=0)
        self.li = Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.c1(x))
        x = torch.relu(self.c2(x))
        x = torch.flatten(x, 1)
        x = self.li(x)
        return x


if __name__ == "__main__":
    x, y = load_digits(return_X_y=True)
    x, x_test, y, y_test = train_test_split(x, y, test_size=0.2)

    x = (x / 8.0).reshape(-1, 1, 8, 8)
    x_test = (x_test / 8.0).reshape(-1, 1, 8, 8)

    # Save for later
    with open(X_TEST_PATH, "wb") as f:
        dump_ndarray(x_test, f)

    with open(Y_TEST_PATH, "wb") as f:
        dump_ndarray(y_test, f)

    x = torch.FloatTensor(x)
    x_test = torch.FloatTensor(x_test)
    y = torch.LongTensor(y)
    y_test = torch.LongTensor(y_test)

    clf = Classifier()
    opt = Adam(params=clf.parameters())
    ce_loss = CrossEntropyLoss()

    batch_size = 100
    max_iter = 200

    final_loss = torch.inf
    for i in range(max_iter):
        losses = []
        for j in range(0, len(x), batch_size):
            x_batch = x[j : j + batch_size]
            y_batch = y[j : j + batch_size]
            opt.zero_grad()
            pred = clf(x_batch)
            loss = ce_loss(pred, y_batch)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        mu_loss = np.mean(losses)

        if i % 20 == 0:
            print(f"loss at iter-{i+1}: {mu_loss}")
        final_loss = mu_loss

    with torch.no_grad():
        clf.eval()
        pred = clf(x_test)
        labels = pred.argmax(1)

        acc = (labels == y_test).float().mean()

        print(f"test accuracy: {acc.item()}")

    print(f"Dumping weights to {WEIGHT_PATH}...")
    with open(WEIGHT_PATH, "wb") as f:
        dump_torch_conv2d(clf.c1, f)
        dump_torch_conv2d(clf.c2, f)
        dump_torch_linear(clf.li, f)
