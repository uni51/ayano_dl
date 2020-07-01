import numpy as np
import math

# 学習データの数
N = 1000

# (学習に再現性をもたせるために シードを固定しています。本来は不要です)
np.random.seed(1)

# 適当な学習データと正解ラベルを生成
TX = (np.random.rand(N, 2) * 1000).astype(np.int32) + 1
TY = (TX.min(axis=1) / TX.max(axis=1) <= 0.2).astype(np.int)[np.newaxis].T

# 平均と標準偏差を計算
MU = TX.mean(axis=0)
SIGMA = TX.std(axis=0)

# 標準化
def standardize(X):
    return (X - MU) / SIGMA

TX = standardize(TX)

# 重みとバイアス
W1 = np.random.randn(2, 2) # 第1層重み
W2 = np.random.randn(2, 2) # 第2層重み
W3 = np.random.randn(1, 2) # 第3層重み
b1 = np.random.randn(2)    # 第1層バイアス
b2 = np.random.randn(2)    # 第2層バイアス
b3 = np.random.randn(1)    # 第3層バイアス

# シグモイド関数
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# 順伝播
def forward(X0):
    Z1 = np.dot(X0, W1.T) + b1
    X1 = sigmoid(Z1)
    Z2 = np.dot(X1, W2.T) + b2
    X2 = sigmoid(Z2)
    Z3 = np.dot(X2, W3.T) + b3
    X3 = sigmoid(Z3)

    return Z1, X1, Z2, X2, Z3, X3

# シグモイド関数の微分
def dsigmoid(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

# 出力層のデルタ
def delta_output(Z, Y):
    return (sigmoid(Z) - Y) * dsigmoid(Z)

# 隠れ層のデルタ
def delta_hidden(Z, D, W):
    return dsigmoid(Z) * np.dot(D, W)

# 逆伝播
def backward(Y, Z3, Z2, Z1):
    D3 = delta_output(Z3, Y)
    D2 = delta_hidden(Z2, D3, W3)
    D1 = delta_hidden(Z1, D2, W2)

    return D3, D2, D1

# 学習率
ETA = 0.001

# 目的関数の重みでの微分
def dweight(D, X):
    return np.dot(D.T, X)

# 目的関数のバイアスでの微分
def dbias(D):
    return D.sum(axis=0)

# パラメータの更新
def update_parameters(D3, X2, D2, X1, D1, X0):
    global W3, W2, W1, b3, b2, b1

    W3 = W3 - ETA * dweight(D3, X2)
    W2 = W2 - ETA * dweight(D2, X1)
    W1 = W1 - ETA * dweight(D1, X0)

    b3 = b3 - ETA * dbias(D3)
    b2 = b2 - ETA * dbias(D2)
    b1 = b1 - ETA * dbias(D1)

# 学習
def train(X, Y):
    # 順伝播
    Z1, X1, Z2, X2, Z3, X3 = forward(X)

    # 逆伝播
    D3, D2, D1 = backward(Y, Z3, Z2, Z1)

    # パラメータの更新
    update_parameters(D3, X2, D2, X1, D1, X)

# 繰り返し回数
EPOCH = 30000

# 予測
def predict(X):
    return forward(X)[-1]

# 目的関数
def E(Y, X):
    return 0.5 * ((Y - predict(X)) ** 2).sum()

# ミニバッチ数
BATCH = 100

for epoch in range(1, EPOCH + 1):
    # ミニバッチ学習用にランダムなインデックスを取得
    p = np.random.permutation(len(TX))

    # ミニバッチの数分だけデータを取り出して学習
    for i in range(math.ceil(len(TX) / BATCH)):
        indice = p[i*BATCH:(i+1)*BATCH]
        X0 = TX[indice]
        Y  = TY[indice]

        train(X0, Y)

    # ログを残す
    if epoch % 1000 == 0:
        log = '誤差 = {:8.4f} ({:5d}エポック目)'
        print(log.format(E(TY, TX), epoch))

# 分類器
def classify(X):
    return (predict(X) > 0.8).astype(np.int)

# テストデータ生成
TEST_N = 1000
testX = (np.random.rand(TEST_N, 2) * 1000).astype(np.int32) + 1
testY = (testX.min(axis=1) / testX.max(axis=1) <= 0.2).astype(np.int)[np.newaxis].T

# 精度計算
accuracy = (classify(standardize(testX)) == testY).sum() / TEST_N
print('精度: {}%'.format(accuracy * 100))
