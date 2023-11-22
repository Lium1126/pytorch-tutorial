# IMPORT ========================================================================================
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


# CONSTANT Parameter ============================================================================
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.05

# Neural Network Definition =====================================================================
class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)


# FUNCTION Definition ===========================================================================
### mnistのデータをファイルから読み込む
def loadData(PATH, FILENAME):
	with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
			((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

	# numpyのvectorからtorchのtensorに変換
	return map(torch.tensor, (x_train, y_train, x_valid, y_valid))


### 学習用DataSetおよび検証用DataSetから、それぞれのDataLoaderをインスタンス化する
def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )


### NNのモデルおよび最適化器をインスタンス化する
def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=LEARNING_RATE)


### 学習および検証で共通する誤差計算を行う
### 必要に応じて、パラメータ更新も行う
def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

	# optがNoneでない場合は学習フェーズであるから、パラメータを更新
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


### 学習および過学習の検証をする
def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
		# 学習フェーズ(Training)
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

		# 検証フェーズ(Validation)
        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print("epoch {:}:".format(epoch), "loss = {:.12f}".format(val_loss))


# Main =========================================================================================
if __name__ == "__main__":
	# mnistからデータを読み込む
	x_train, y_train, x_valid, y_valid = loadData(Path("data/mnist"), "mnist.pkl.gz")

	# 読み込んだデータからDataSetを作成
	train_ds = TensorDataset(x_train, y_train)
	valid_ds = TensorDataset(x_valid, y_valid)

	# DataLoaderの取得からモデルの最適化までが3行となった
	train_dl, valid_dl = get_data(train_ds, valid_ds, BATCH_SIZE)
	model, opt = get_model()
	fit(EPOCHS, model, F.cross_entropy, opt, train_dl, valid_dl)
