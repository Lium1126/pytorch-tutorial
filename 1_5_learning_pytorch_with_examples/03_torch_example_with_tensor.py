# -*- coding: utf-8 -*-
import torch


DTYPE = torch.float
# 以下のいずれかを選択
DEVICE = torch.device("cpu")
# DEVICE = torch.device("cuda:0")

N = 64       # N: バッチサイズ
D_in = 1000  # D_in: 入力層の次元数
H = 100      # H: 隠れ層の次元数
D_out = 10   # D_out: 出力層の次元数

# 乱数により入力データと目標となる出力データを生成
x = torch.randn(N, D_in, device=DEVICE, dtype=DTYPE)
y = torch.randn(N, D_out, device=DEVICE, dtype=DTYPE)

# 乱数による重みの初期化
w1 = torch.randn(D_in, H, device=DEVICE, dtype=DTYPE)
w2 = torch.randn(H, D_out, device=DEVICE, dtype=DTYPE)

learning_rate = 1e-6
for t in range(500):
    # 順伝播： 予測値yの計算
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # 損失の計算と表示
    # 損失は形状[1,]のTensorになります。
    # loss.item() は損失を表すTensorの持つ値をスカラー値で返します
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print('epoch {:0>3}:'.format(t), 'loss = {:.12f}'.format(loss))

   # 逆伝搬：損失に対するW1とw2の勾配の計算
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # 確率的勾配降下法による重みの更新
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2