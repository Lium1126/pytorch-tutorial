# -*- coding: utf-8 -*-
import torch


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        forward関数は入力データのTensorを受け入れ、出力データのTensorを返します。
        Tensorの任意の演算子と同様に、コンストラクタで定義されたモジュールを使用できます。
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


N = 64       # N: バッチサイズ
D_in = 1000  # D_in: 入力層の次元数
H = 100      # H: 隠れ層の次元数
D_out = 10   # D_out: 出力層の次元数

# 乱数により入力データと目標となる出力データを表すTensorを生成
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# 上で定義したクラスをインスタンス化してモデルを構築
model = TwoLayerNet(D_in, H, D_out)

# 損失関数とオプティマイザを定義します。
# model.parameters()を呼び出すことで、モデルのメンバ変数である2つのnn.Linearモジュールの
# 学習可能なパラメータをSGDのコンストラクタの引数として渡すことができます。
loss_func = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

for t in range(500):
    # 順伝播:入力xから予測値yをモデルで算出します。
    y_pred = model(x)

    # 損失の計算と表示
    loss = loss_func(y_pred, y)
    if t % 100 == 99:
        print('epoch {:0>3}:'.format(t), 'loss = {:.12f}'.format(loss.item()))

    # 勾配を0に初期化し、逆伝播を実行することで重みを更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()