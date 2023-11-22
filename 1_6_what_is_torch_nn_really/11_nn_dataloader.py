from pathlib import Path
import pickle
import gzip
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


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
train_ds = TensorDataset(x_train, y_train) # Datasetを利用
train_dl = DataLoader(train_ds, batch_size=bs) # DataLoaderをインスタンス化
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
        # =====================================================
        # 以前はミニバッチの取得のために面倒な計算が必要だった
        # =====================================================
        # for i in range((n - 1) // bs + 1):
        #     xb, yb = train_ds[i * bs: i * bs + bs]

        # =====================================================
        # DataLoaderを利用することで簡単にミニバッチを取得可能になった
        # =====================================================
        for xb, yb in train_dl:

            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            opt.step()
            opt.zero_grad()


fit()

print("after learning----------------------------------------------")
print("loss:", loss_func(model(xb), yb))