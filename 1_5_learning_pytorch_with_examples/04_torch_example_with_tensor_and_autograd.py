# -*- coding: utf-8 -*-
import torch

DTYPE = torch.float
DEVICE = torch.device("cpu")

N = 64       # N: バッチサイズ
D_in = 1000  # D_in: 入力層の次元数
H = 100      # H: 隠れ層の次元数
D_out = 10   # D_out: 出力層の次元数

# 逆伝播の際にこのTensorに対する勾配を計算する必要がない場合は、requires_grad=Falseを指定します(デフォルト)
x = torch.randn(N, D_in, device=DEVICE, dtype=DTYPE)
y = torch.randn(N, D_out, device=DEVICE, dtype=DTYPE)

# 逆伝播の際、このTensorに対する勾配を計算する場合は、requires_grad=Trueを指定します
w1 = torch.randn(D_in, H, device=DEVICE, dtype=DTYPE, requires_grad=True)
w2 = torch.randn(H, D_out, device=DEVICE, dtype=DTYPE, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # 順伝播： Tensorの演算を利用して予測結果yの算出
    # ここはTensorsを使用した順伝播の計算と全く同じ操作ですが、逆伝播の経路を手動で
    # 定義していないので、順伝播の途中の値を保持しておく必要はありません。
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # Tensorを用いた損失の計算と表示
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print('epoch {:0>3}:'.format(t), 'loss = {:.12f}'.format(loss))

    # autograd を使用して逆伝播の計算をします。
    # backward()によりrequires_gradにTrueが設定されているすべてのTensorに対して、
    # 損失の勾配を計算を行います。これにより、w1.gradとw2.gradはそれぞれw1とw2に
    # 対する損失の勾配を保持するTensorになります。
    loss.backward()

    # 確率的勾配降下法を使って手動で重みを更新します。
    # 重みには requires_gradにTrue が設定されていますが、ここではautogradによる
    # 追跡を避ける必要があるので、torch.no_grad()のブロック内で実行します。
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # 重みの更新後、手動で勾配を0に初期化
        w1.grad.zero_()
        w2.grad.zero_()