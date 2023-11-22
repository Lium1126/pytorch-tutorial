import torch

N = 64       # N: バッチサイズ
D_in = 1000  # D_in: 入力層の次元数
H = 100      # H: 隠れ層の次元数
D_out = 10   # D_out: 出力層の次元数

# 乱数により入力データと目標となる出力データを表すTensorを生成
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# nnパッケージを用いてモデルと損失関数を定義
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
loss_fn = torch.nn.MSELoss(reduction='sum')

# optimパッケージを使用して、モデルの重みを更新するオプティマイザを定義します。
# ここではAdamを使用します。optimパッケージには他にも多くの最適化アルゴリズムが存在ます。
# Adamのコンストラクタの最初の引数により、オプティマイザがどのTensorを更新するか指定できます。
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(500):
    # 順伝播:入力xから予測値yをモデルで算出します。。
    y_pred = model(x)

    # 損失の計算と表示
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print('epoch {:0>3}:'.format(t), 'loss = {:.12f}'.format(loss.item()))

    # 逆伝播に入る前に、更新されることになる変数（モデルの学習可能な重み）の勾配を
    # optimaizerを使用して0に初期化します。
    # これは、デフォルトで.backward()が呼び出される度に勾配がバッファに蓄積されるため
    # 必要になる操作です（上書きされるわけではない）。
    # 詳しくはtorch.autograd.backwardのドキュメントを参照してください。
    optimizer.zero_grad()

    # 逆伝播：モデルのパラメータに対応する損失の勾配を計算
    loss.backward()

    # オプティマイザのstep関数を呼び出すことでパラメータを更新
    optimizer.step()
