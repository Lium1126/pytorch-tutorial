# -*- coding: utf-8 -*-
import torch

N = 64       # N: バッチサイズ
D_in = 1000  # D_in: 入力層の次元数
H = 100      # H: 隠れ層の次元数
D_out = 10   # D_out: 出力層の次元数

# 乱数により入力データと目標となる出力データを表すTensorを生成
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# nnパッケージを利用し、レイヤーの連なりとしてモデルを定義します。
# nn.Sequentialは他のモジュールを並べて保持することで、それぞれのモジュールを順番に
# 実行し、出力を得ます。各Linearモジュールは線形関数を使用して入力から出力を計算し、
# 重みとバイアスを内部のTensorで保持します。
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

# nnパッケージには、一般的な損失関数が含まれています。
# 今回は損失関数として平均二乗誤差（MSE）を使用します。
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
for t in range(500):
    # 順伝播: モデルを流れる入力xから予測値yを計算します。
    # Pythonのモジュールオブジェクトは__call__演算子をオーバーライドするため、
    # 関数のように呼び出すことができます。これにより、入力データのTensorを
    # モジュールに渡すことで、出力データのTensorを得ることができます。
    y_pred = model(x)

    # 損失の計算と表示
    # 損失関数にyの予測値と正解の値を持つTensorを渡すことで損失を持つTenosrを
    # 得ることができます。
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print('epoch {:0>3}:'.format(t), 'loss = {:.12f}'.format(loss.item()))

    # 逆伝播の前に勾配を0に初期化
    model.zero_grad()

    # 逆伝播: モデルの学習可能なすべてのパラメータに対して損失の勾配を計算
    # 内部では、requires_grad=TrueとなっているすべてのTensorにそれぞれのモデルのパラメータが
    # 保持されているので、モデルが持つ学習可能なパラメータの勾配をすべて計算することできます。
    loss.backward()

    # 確率的勾配降下法を用いた重みの更新
    # 各々のパラメータはTensorなので、これまでと同じ方法で勾配を参照することができます。
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad