# AND, OR, XOR 연산을 하는 네트워크처럼 여기서도 딥러닝 네트워크를 만들 수 있다.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = [0.3, -0.78, 1.26, 0.03, 1.11, 0.24, -0.24,
 -0.47, -0.77, -0.37, -0.85, -0.41, -0.27, 0.02, -0.76, 2.66]
Y = [12.27, 14.44, 11.87, 18.75, 17.52, 16.37,
 19.78, 19.51, 12.65, 14.74, 10.72, 21.94, 12.83, 15.51, 17.14, 14.42]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=6, activation='tanh', input_shape=(1,)),
    # tanh는 하이퍼볼릭 탄젠트 함수로 삼각함수 중 탄젠트 함수와 연관이 있으며
    # -1과 1 사이의 출력을 반환한다.
    tf.keras.layers.Dense(units=1)
])

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1), loss='mse')
model.summary()

history = model.fit(X, Y, epochs=10)
#model.predict(X)

# 회귀 선 그리기
line_x = np.arange(min(X), max(X), 0.01)
line_y = model.predict(line_x)

# 그래프를 그리면 2차함수와 비슷한 곡선이 나온다.   
# 차이점은 딥러닝 네트워크가 예측한 회귀선은 좀 더 직선에 가까운 완만한 형태
plt.plot(line_x, line_y, 'r-')
plt.plot(X, Y, 'bo')
plt.xlabel('Population Growth Rate (%)')
plt.ylabel('Elderly Population Rate (%)')
plt.show()
