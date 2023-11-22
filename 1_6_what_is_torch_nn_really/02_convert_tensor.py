from pathlib import Path
import pickle
import gzip
import numpy as np
import torch

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"
FILENAME = "mnist.pkl.gz"

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

# numpyのvectorからtorchのtensorに変換
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
print("x_train:", x_train, '\n')
print("y_train:", y_train, '\n')
print("x_train.shape:", x_train.shape)
print("y_train.min:", y_train.min().item(), "y_train.max:", y_train.max().item())