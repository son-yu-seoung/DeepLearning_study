import tensorflow as tf
# tensorflow.keras에서 boston_housing 데이터 셋 불러오는 방법
from tensorflow.keras.datasets import boston_housing
(train_X, train_Y), (test_X, test_Y) = boston_housing.load_data()

# train_X(404, 13), train_Y(404,) 13개의 항목을 조합하여 1개의 집 값이 나오는 데이터
# test_X(102, 13), test_Y(102,) train과 test의 비율 404 : 102 약 80 : 20 비율
print(train_X.shape)
print(train_Y.shape)
 
# 보스턴 주택 가격 데이터셋의 데이터 속성은 각 데이터의 단위가 다르다. (비율, 0/1, 양의 정수 등)
# 실생활에서 얻는 데이터는 이처럼 다양한 단위를 가지고 있는 경우가 많다.
# 딥러닝에서 이러한 데이터를 전처리해서 정규화(Standardization)를 해야 학습 효율이 좋다.
x_mean = train_X.mean(axis=0) # 0 : 세로, 1 : 가로(기준)
x_std = train_X.std(axis=0) # std = 표준편차
train_X -= x_mean
train_X /= x_std
test_X -= x_mean
test_X /= x_std

y_mean = train_Y.mean(axis=0)
y_std = train_Y.std(axis=0)
train_Y -= y_mean
train_Y /= y_std
test_Y -= y_mean
test_Y /= y_std

print(train_X[0], train_Y[0])

# 딥러닝 네트워크 학습
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=52, activation='relu', input_shape=(13,)),
    tf.keras.layers.Dense(units=39, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.07), loss='mse')
model.summary()
# # model.fit으로 회귀 모델 학습, validation=0.25는 train 데이터 중 검증 데이터의 비율
# history = model.fit(train_X, train_Y, epochs=25, batch_size=32, validation_split=0.25)

# 네트워크가 훈련 데이터에 과적합되지 않도록 콜백(callback) 함수 사용 
# 이렇게 과적합을 미리 방지하니 산점도 그래프가 조금 더 안정되었음 (프로젝트 때 사용 해보기!)
history = model.fit(train_X, train_Y, epochs=25, batch_size=32, validation_split=0.25,
 callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss')])

# 시각화
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# 테스트 데이터를 이용해 회귀 모델을 평가
model.evaluate(test_X, test_Y)

# 실제 주택 가격과 예측 주택 가격을 1:1로 비교
pred_Y = model.predict(test_X)

plt.figure(figsize=(5,5))
plt.plot(test_X, pred_Y, 'b.')

plt.plot([min(test_Y), max(test_Y)], [min(test_Y), max(test_Y)]) # y=x
plt.xlabel('test_Y')
plt.ylabel('pred_Y')
plt.show()





















