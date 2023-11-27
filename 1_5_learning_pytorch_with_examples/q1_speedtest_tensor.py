# -*- coding: utf-8 -*-
import numpy as np
import torch
import time


DTYPE = torch.float
DEVICE = torch.device("cpu")
# DEVICE = torch.device("cuda:0")



TEST_NUM = 1000
THRESHOLD = 1e-6
EPOCHS = 1000
N = 64       # N: バッチサイズ
D_in = 1000  # D_in: 入力層の次元数
H = 100      # H: 隠れ層の次元数
D_out = 10   # D_out: 出力層の次元数

with open('./tensor_same_data.csv', mode='w') as f:
    f.write('epoch,learning_times,real_time\n')

    with open('./input.txt', mode='r') as input_file:
        for i in range(1, TEST_NUM+1):
            x = [list(map(float, input_file.readline().split())) for _ in range(N)]
            x = np.array(x)
            y = [list(map(float, input_file.readline().split())) for _ in range(N)]
            y = np.array(y)
            w1 = [list(map(float, input_file.readline().split())) for _ in range(D_in)]
            w1 = np.array(w1)
            w2 = [list(map(float, input_file.readline().split())) for _ in range(H)]
            w2 = np.array(w2)

            # # 乱数により入力データと目標となる出力データを生成
            # x = np.random.randn(N, D_in)
            # y = np.random.randn(N, D_out)

            # # 乱数による重みの初期化
            # w1 = np.random.randn(D_in, H)
            # w2 = np.random.randn(H, D_out)

            x, y, w1, w2 = map(torch.tensor, (x, y, w1, w2))

            learning_rate = 1e-6
            time_start = time.time()
            for epoch in range(EPOCHS):
                # 順伝播： 予測値yの計算
                h = x.mm(w1)
                h_relu = h.clamp(min=0)
                y_pred = h_relu.mm(w2)

                # 損失の計算と表示
                loss = (y_pred - y).pow(2).sum().item()
                if loss <= THRESHOLD:
                    time_end = time.time()
                    f.write(f'{i},{epoch},{time_end - time_start}\n')
                    break

                # 逆伝搬：損失に対するW1とw2の勾配の計算
                grad_y_pred = 2.0 * (y_pred - y)
                grad_w2 = h_relu.t().mm(grad_y_pred)
                grad_h_relu = grad_y_pred.mm(w2.t())
                grad_h = grad_h_relu.clone()
                grad_h[h < 0] = 0
                grad_w1 = x.t().mm(grad_h)

                # 重みの更新
                w1 -= learning_rate * grad_w1
                w2 -= learning_rate * grad_w2

                if epoch == EPOCHS - 1:
                    time_end = time.time()
                    f.write(f'{i},-1,{time_end - time_start}\n')
            
            print(f'Test {i} finished')