from pathlib import Path
import pickle
import gzip
import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F

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
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, xb):
        return xb @ self.weights + self.bias


model = Mnist_Logistic()
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

			# ===================================================
			# 以前は重みの更新処理で変数名を意識しなければならなかった
			# ===================================================
			# with torch.no_grad():
			# 	weights -= weights.grad * lr
			# 	bias -= bias.grad * lr
			# 	weights.grad.zero_()
			# 	bias.grad.zero_()

			# ===================================================
			# nn.Moduleを使うことで変数名に依存しない実装になっている
			# ===================================================
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= p.grad * lr
                model.zero_grad()


fit()

print("after learning----------------------------------------------")
print("loss:", loss_func(model(xb), yb))