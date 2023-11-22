from pathlib import Path
import pickle
import gzip
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset

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

# =====================================================
# 以前は入力と出力のデータをそれぞれミニバッチごとに取得していた
# =====================================================
# xb = x_train[start_i:end_i]
# yb = y_train[start_i:end_i]

# =====================================================
# Datasetを使用すると入力・出力データを一括で取得できる
# =====================================================
train_ds = TensorDataset(x_train, y_train) # Datasetを利用

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

xb, yb = train_ds[0: bs]
print("before learning---------------------------------------------")
print("loss:", loss_func(model(xb), yb), '\n')

def fit():
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            # =====================================================
            # 以前は次のミニバッチを取得するために面倒な計算が必要だった
            # =====================================================
            # start_i = i * bs
            # end_i = start_i + bs
            # xb = x_train[start_i:end_i]
            # yb = y_train[start_i:end_i]

            # =====================================================
            # Datasetを利用することで簡単にミニバッチを取得できる
            # =====================================================
            xb, yb = train_ds[i * bs: i * bs + bs]

            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            opt.step()
            opt.zero_grad()


fit()

print("after learning----------------------------------------------")
print("loss:", loss_func(model(xb), yb))