import numpy as np


TEST_NUM = 1000
N = 64       # N: バッチサイズ
D_in = 1000  # D_in: 入力層の次元数
H = 100      # H: 隠れ層の次元数
D_out = 10   # D_out: 出力層の次元数

with open('./input.txt', mode='w') as f:
	for cnt in range(TEST_NUM):
		x = np.random.randn(N, D_in)
		y = np.random.randn(N, D_out)
		w1 = np.random.randn(D_in, H)
		w2 = np.random.randn(H, D_out)

		for i in range(N):
			for j in range(D_in):
				if j == D_in - 1:
					f.write(f'{x[i][j]}\n')
				else:
					f.write(f'{x[i][j]} ')
		
		for i in range(N):
			for j in range(D_out):
				if j == D_out - 1:
					f.write(f'{y[i][j]}\n')
				else:
					f.write(f'{y[i][j]} ')

		for i in range(D_in):
			for j in range(H):
				if j == H - 1:
					f.write(f'{w1[i][j]}\n')
				else:
					f.write(f'{w1[i][j]} ')

		for i in range(H):
			for j in range(D_out):
				if j == D_out - 1:
					f.write(f'{w2[i][j]}\n')
				else:
					f.write(f'{w2[i][j]} ')

		print(f'Test case {cnt} finish')