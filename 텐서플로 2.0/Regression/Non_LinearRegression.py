# 머신러닝에서 회귀는 가격이나 확률처럼 연속된 실숫값을 정확히 예측하는 것이 목적입니다.

# 비선형 회귀(Non_Linear Regression)
# 선형 회귀로는 표현할 수 없는 데이터의 경향성을 설명하기 위한 회귀

# 2018년 지역별 인구증가율과 고령인구비율
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
X = [0.3, -0.78, 1.26, 0.03, 1.11, 0.24, -0.24,
 -0.47, -0.77, -0.37, -0.85, -0.41, -0.27, 0.02, -0.76, 2.66]
Y = [12.27, 14.44, 11.87, 18.75, 17.52, 16.37,
 19.78, 19.51, 12.65, 14.74, 10.72, 21.94, 12.83, 15.51, 17.14, 14.42]

# # version 1 (이차곡선)
# # 어려울 것 없다. 단순한 이차방정식 구하기 
# a = tf.Variable(np.random.random())
# b = tf.Variable(np.random.random())
# c = tf.Variable(np.random.random())

# def compute_loss():
#     y_pred = a * X * X + b * X + c
#     loss = tf.reduce_mean((Y - y_pred) ** 2)
#     return loss

# optimizer = tf.optimizers.Adam(lr=0.07)
# for i in range(1000): # 1000번의 학습
#     optimizer.minimize(compute_loss, var_list=[a, b, c]) # 최소화할 손실, 학습할 변수 리스트

#     if i % 100 == 0:
#         # .numpy()를 붙이는 이유 : tensorflow를 Session run하지 않으면 이상한 값 나옴
#         print(i, 'a:', a.numpy(), 'b:', b.numpy(), 'c:', c.numpy(), 'loss:', compute_loss().numpy())

# # 그래프를 그리기 위해 회귀선의 x, y 데이터를 구한다.
#line_x = np.arange(min(X), max(X), 0.01) 
#line_y = a * line_x * line_x + b * line_x + c 

# # version 2 (삼차곡선)
# # 어려울 것 없다. 단순한 삼차방정식 구하기 
# a = tf.Variable(np.random.random())
# b = tf.Variable(np.random.random())
# c = tf.Variable(np.random.random())
# d = tf.Variable(np.random.random())

# def compute_loss():
#     y_pred = a * X * X * X + b * X * X + c * X + d
#     loss = tf.reduce_mean((Y - y_pred) ** 2)
#     return loss

# optimizer = tf.optimizers.Adam(lr=0.07)
# for i in range(1000): # 1000번의 학습
#     optimizer.minimize(compute_loss, var_list=[a, b, c, d]) # 최소화할 손실, 학습할 변수 리스트

#     if i % 100 == 0:
#         # .numpy()를 붙이는 이유 : tensorflow를 Session run하지 않으면 이상한 값 나옴
#         print(i, 'a:', a.numpy(), 'b:', b.numpy(), 'c:', c.numpy(), 'd:', d.numpy(),
#          'loss:', compute_loss().numpy())

# # 그래프를 그리기 위해 회귀선의 x, y 데이터를 구한다.
#line_x = np.arange(min(X), max(X), 0.01)
#line_y = a * line_x * line_x * line_x + b * line_x * line_x + c * line_x + d

plt.plot(X, Y, 'bo')
plt.plot(line_x, line_y, 'r-')
plt.xlabel('Population Growth Rate(%)')
plt.ylabel('Elderly Population Rate(%)')
plt.show()