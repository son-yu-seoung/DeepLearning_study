# Q. 고려해야할 실수의 범위가 100개이고 그중에서 마킹된 두 개의 숫자만 곱해야한다.

# 우선 SimpleRNN으로 문제 먼저 풀이
# 2,560개를 훈련 데이터와 검증 데이터로 사용하고 나머지 440개는 테스트 데이터로 이용
import numpy as np
import tensorflow as tf

x = []
y = []
for i in range(3000):
    # 0 ~ 1 사이의 랜덤한 숫자 100개를 만든다.
    lst = np.random.rand(100)
    # 마킹할 숫자 2개의 인덱스를 뽑는다.
    idx = np.random.choice(100, 2, replace=False)
    # 마킹 인덱스가 저장된 원-핫 인코딩 벡터를 만든다.
    zeros = np.zeros(100)
    zeros[idx] = 1
    # 마킹 인덱스와 랜덤한 숫자를 합쳐서 x에 저장
    x.append(np.array(list(zip(zeros, lst))))
    # 마킹 인덱스가 1인 값만 서로 곱해서 y에 저장
    y.append(np.prod(lst[idx])) # np.prod는 axis를 기준으로 array 내부 elements들의 곱

print(x[0], y[0])

# # RNN 레이어를 겹치기 위해 첫 번째 SimpleRNN 레이어에서 return_sequences=Ture로 설정
# # return_sequences는 레이어의 출력을 다음 레이어로 그대로 넘겨주게 된다.
# model = tf.keras.Sequential([
#     tf.keras.layers.SimpleRNN(30, return_sequences=True, input_shape=[100, 2]),
#     tf.keras.layers.SimpleRNN(30),
#     tf.keras.layers.Dense(1)
# ])

# # LSTM
# model = tf.keras.Sequential([
#     tf.keras.layers.LSTM(30, return_sequences=True, input_shape=[100, 2]),
#     tf.keras.layers.LSTM(30),
#     tf.keras.layers.Dense(1)
# ])

# GRU
# GRU 레이어는 LSTM 레이어와 비슷한 역할을 하지만 구조가 더 간단하기 때문에 계산상의 이점이 있다.
# 어떤 문제에서는 LSTM보다 GRU가 더 좋다.
# GRU를 사용하면 파라미터의 수가 11,311 -> 8,671개로 줄어든다.
model = tf.keras.Sequential([
    tf.keras.layers.GRU(30, return_sequences=True, input_shape=[100, 2]),
    tf.keras.layers.GRU(30),
    tf.keras.layers.Dense(1)
])



model.compile(optimizer='adam', loss='mse')
model.summary()

X = np.array(x)
Y = np.array(y)
# 2560개의 데이터만 학습시키비다. 검증 데이터는 20%로 지정한다.
history = model.fit(X[:2560], Y[:2560], epochs=100, validation_split=0.2)

import matplotlib.pyplot as plt

plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()