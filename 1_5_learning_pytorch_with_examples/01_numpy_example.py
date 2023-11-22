# -*- coding: utf-8 -*-
import numpy as np

N = 64       # N: バッチサイズ
D_in = 1000  # D_in: 入力層の次元数
H = 100      # H: 隠れ層の次元数
D_out = 10   # D_out: 出力層の次元数

# 乱数により入力データと目標となる出力データを生成
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# 乱数による重みの初期化
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6
for t in range(500):
    # 順伝播： 予測値yの計算
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # 損失の計算と表示
    loss = np.square(y_pred - y).sum()
    if t % 100 == 99:
        print('epoch {:0>3}:'.format(t), 'loss = {:.12f}'.format(loss))

    # 逆伝搬：損失に対するW1とw2の勾配の計算
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # 重みの更新
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2