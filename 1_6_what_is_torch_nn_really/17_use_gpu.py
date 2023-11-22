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
LEARNING_RATE = 0.1
# 使用可能であればCUDA環境を利用する
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# DataLoader Wrapping ===========================================================================
class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def preprocess(x, y):
    return x.view(-1, 1, 28, 28).to(DEVICE), y.to(DEVICE)


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))


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
    model = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        Lambda(lambda x: x.view(x.size(0), -1)),
    )
    model.to(DEVICE)
    return model, optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)


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
    train_dl = WrappedDataLoader(train_dl, preprocess)
    valid_dl = WrappedDataLoader(valid_dl, preprocess)
    model, opt = get_model()
    fit(EPOCHS, model, F.cross_entropy, opt, train_dl, valid_dl)
