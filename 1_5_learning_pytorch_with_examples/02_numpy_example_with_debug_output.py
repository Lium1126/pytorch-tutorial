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

#学習前の重みを定義（前の学習で使用された重みの初期値とは異なる）
w1_unlearned = np.random.randn(D_in, H)
w2_unlearned = np.random.randn(H, D_out)

#学習前の出力を算出
h = x.dot(w1_unlearned)
h_relu = np.maximum(h, 0)
y_pred = h_relu.dot(w2_unlearned)
print(f"学習前出力：{np.round(y_pred[0], decimals=2)}")

#学習によって得られた重みで出力を算出
h = x.dot(w1)
h_relu = np.maximum(h, 0)
y_pred = h_relu.dot(w2)
print(f"学習後出力：{np.round(y_pred[0], decimals=2)}")

#目的の出力yとの比較
print(f"目的の出力：{np.round(y[0], decimals=2)}")
