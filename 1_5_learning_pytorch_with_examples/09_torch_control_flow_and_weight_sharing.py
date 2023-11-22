# -*- coding: utf-8 -*-
import random
import torch


class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        # コンストラクタで順伝播の処理を定義
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        順伝播の経路の実装では、ランダムに0から4までの値を選択し、
        その数だけ中間層のモジュールを再利用することで、隠れ層の出力を計算します。

        それぞれの順伝播の経路を表す計算グラフは動的に変化するので、
        繰り返しや、条件分岐といったPythonの標準的なフロー制御を利用して
        順伝播の経路を定義することができます。

        この結果から確認できるように、計算グラフを定義する際に、問題なく何度も同じ
        モジュールを使いまわすことができるのは、一度しかモジュールを利用することが
        できなったLua Torchから大きく改善したところと言えます。
        """
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0, 4)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred


N = 64       # N: バッチサイズ
D_in = 1000  # D_in: 入力層の次元数
H = 100      # H: 隠れ層の次元数
D_out = 10   # D_out: 出力層の次元数

# 乱数により入力データと目標となる出力データを表すTensorを生成
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# 上で定義したクラスをインスタンス化してモデルを構築します。
model = DynamicNet(D_in, H, D_out)

# 損失関数とオプティマイザを定義します。
# この奇妙なモデルを通常の確率勾配降下法で訓練するのは難しいので、モーメンタムを使用します。
loss_func = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

for t in range(500):
    # 順伝播:入力xから予測値yをモデルで算出します。
    y_pred = model(x)

    # 損失の計算と表示
    loss = loss_func(y_pred, y)
    if t % 100 == 99:
        print('epoch {:0>3}:'.format(t), 'loss = {:.12f}'.format(loss.item()))

    # 勾配を0に初期化し、逆伝播を実行することで重みを更新します
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
