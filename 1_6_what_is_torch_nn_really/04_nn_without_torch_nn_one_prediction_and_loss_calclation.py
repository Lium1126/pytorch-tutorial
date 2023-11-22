from pathlib import Path
import pickle
import gzip
import numpy as np
import torch
import math

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"
FILENAME = "mnist.pkl.gz"

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

# numpyのvectorからtorchのtensorに変換
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)

# 重みの初期化
weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_() # <-------------------- 初期化作業追跡されないよう、初期化した後にrequires_gradを追加
bias = torch.zeros(10, requires_grad=True)

# 損失関数と活性化関数を定義(python標準の関数宣言として)
def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
    return log_softmax(xb @ weights + bias) # @: 行列積


bs = 64  # batch size
xb = x_train[0:bs]  # a mini-batch from x
preds = model(xb)  # predictions

# 負の対数尤度関数を定義(pythonの標準関数宣言)
def nll(input, target):
    return -input[range(target.shape[0]), target].mean()

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


loss_func = nll

yb = y_train[0:bs]
print("loss:", loss_func(preds, yb))
print("accuracy:", accuracy(preds, yb))