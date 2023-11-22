# -*- coding: utf-8 -*-
import torch


class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


DTYPE = torch.float
DEVICE = torch.device("cpu")

N = 64       # N: バッチサイズ
D_in = 1000  # D_in: 入力層の次元数
H = 100      # H: 隠れ層の次元数
D_out = 10   # D_out: 出力層の次元数

# 乱数により入力データと目標となる出力データを表すTensorを生成
x = torch.randn(N, D_in, device=DEVICE, dtype=DTYPE)
y = torch.randn(N, D_out, device=DEVICE, dtype=DTYPE)

# 乱数による重みを表すTensorの定義
w1 = torch.randn(D_in, H, device=DEVICE, dtype=DTYPE, requires_grad=True)
w2 = torch.randn(H, D_out, device=DEVICE, dtype=DTYPE, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # 関数を適用するには、Function.applyメソッドを用います。
    # reluと命名しておきます。
    relu = MyReLU.apply

    # 順伝播：独自のautograd操作を用いてReLUの出力を算出することで予想結果yを計算します。
    y_pred = relu(x.mm(w1)).mm(w2)

    # 損失の計算と表示
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print('epoch {:0>3}:'.format(t), 'loss = {:.12f}'.format(loss.item()))

    # autogradを利用して逆伝播を実施
    loss.backward()

    # 確率的勾配降下法を用いた重みの更新
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # 重みの更新後、手動で勾配を0に初期化
        w1.grad.zero_()
        w2.grad.zero_()