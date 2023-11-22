from pathlib import Path
import pickle
import gzip
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"
FILENAME = "mnist.pkl.gz"

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

# numpyのvectorからtorchのtensorに変換
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape
bs = 64  # batch size
xb = x_train[0:bs]  # a mini-batch from x
yb = y_train[0:bs]
epochs = 10
lr = 0.05  # learning rate

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)


def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=lr)


model, opt = get_model()
loss_func = F.cross_entropy

print("before learning---------------------------------------------")
print("loss:", loss_func(model(xb), yb), '\n')

def fit():
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()

            # =====================================================
            # 以前は手動でパラメータの更新をしていた
            # =====================================================
            # with torch.no_grad():
            #     for p in model.parameters():
            #         p -= p.grad * lr
            #     model.zero_grad()

            # =====================================================
            # パラメータ更新を手動でやらずとも、自動でやってくれる
            # =====================================================
            opt.step()
            opt.zero_grad()


fit()

print("after learning----------------------------------------------")
print("loss:", loss_func(model(xb), yb))