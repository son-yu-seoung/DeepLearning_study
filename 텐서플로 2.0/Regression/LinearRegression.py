# 머신러닝에서 회귀는 가격이나 확률처럼 연속된 실숫값을 정확히 예측하는 것이 목적입니다.

# 선형 회귀(Linear Regression)
# 데이터의 경향성을 가장 잘 설명하는 하나의 직선을 예측하는 것

# 2018년 지역별 인구증가율과 고령인구비율
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
X = [0.3, -0.78, 1.26, 0.03, 1.11, 0.24, -0.24,
 -0.47, -0.77, -0.37, -0.85, -0.41, -0.27, 0.02, -0.76, 2.66]
Y = [12.27, 14.44, 11.87, 18.75, 17.52, 16.37,
 19.78, 19.51, 12.65, 14.74, 10.72, 21.94, 12.83, 15.51, 17.14, 14.42]

# # version 1 ( no tensorflow ) 
# # X, Y의 평균을 구한다.
# x_bar = sum(X) / len(X)
# y_bar = sum(Y) / len(Y)

# # 최소 제곱법으로 a, b를 구한다. (직선 구하기)
# a = sum([(y - y_bar) * (x - x_bar) for y, x in list(zip(Y,X))])
# a /= sum([(x - x_bar) ** 2 for x in X])
# b = y_bar - a * x_bar
# print('a :', a, 'b :', b)

# # version 2 ( tensorflow )
# a = tf.Variable(np.random.random())
# b = tf.Variable(np.random.random())

# def compute_loss():
#     y_pred = a * X + b
#     loss = tf.reduce_mean((Y - y_pred) ** 2)
#     return loss

# optimizer = tf.optimizers.Adam(lr=0.07)
# for i in range(1000): # 1000번의 학습
#     optimizer.minimize(compute_loss, var_list=[a, b]) # 최소화할 손실, 학습할 변수 리스트

#     if i % 100 == 0:
#         # .numpy()를 붙이는 이유 : tensorflow를 Session run하지 않으면 이상한 값 나옴
#         print(i, 'a:', a.numpy(), 'b:', b.numpy(), 'loss:', compute_loss().numpy())

# 그래프를 그리기 위해 회귀선의 x, y 데이터를 구한다.
line_x = np.arange(min(X), max(X), 0.01) # 중요 그래프 그릴 때 x 범위 잡기!
line_y = a * line_x + b

plt.plot(X, Y, 'bo')
plt.plot(line_x, line_y, 'r-')
plt.xlabel('Population Growth Rate(%)')
plt.ylabel('Elderly Population Rate(%)')
plt.show()