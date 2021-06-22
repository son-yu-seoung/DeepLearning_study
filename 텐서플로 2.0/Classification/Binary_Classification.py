# 분류(Classification)이란 가장 기초적인 데이터 분석 방법 중 하나
# 데이터가 어느 범주(Category)에 해당하는지 판단하는 문제

# 와인 데이터셋은 보스턴 주택 데이터세과 달리 외부에서 데이터를 불러오고 정제해야 함
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
red = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')
white = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', sep=';')
# 12개의 속성으로 구성되어 있음
print(red.head())
print(white.head())

# + 와인이 레드 와인인지 화이트 와인인지 표시해주는 속성을 추가한 후 레드, 화이트 와인 데이터를 합쳐야 함
red['type'] = 0
white['type'] = 1
wine = pd.concat([red, white])
print(wine.describe)

# 히스토그램으로 red, white 와인의 개수를 시각화
plt.hist(wine['type'])
plt.xticks([0, 1])
plt.show() 
# 1   4898, 0   1599
print(wine['type'].value_counts())# white wine의 개수가 약 3배 정도는 더 많다.

# info() 함수는 데이터프레임을 구성하는 속성들 의 정보를 알려줌
# 정규화 과정에서 데이터에 숫자가 아닌 값이 들어가면 에러의 원인이 되기 때문에 확인 필요
print(wine.info())

# 각 데이터마다 단위가 다르기 때문에 정규화 과정 필요
wine_norm = (wine - wine.min()) / (wine.max() - wine.min())
print(wine_norm.head())
print(wine_norm.describe)

# 정규화된 데이터를 랜덤하게 섞고 학습을 위해 numpy로 변환하기 
wine_shuffle = wine_norm.sample(frac=1) # 전체 데이터 중 frac의 비율만큼(100%)
wine_np = wine_shuffle.to_numpy()

# X, Y 분리, 학습 데이터와 테스트 데이터 분리
import tensorflow as tf
train_idx = int(len(wine_np) * 0.8)
train_X, train_Y = wine_np[:train_idx, :-1], wine_np[:train_idx, -1]
test_X, test_Y = wine_np[train_idx:, :-1], wine_np[train_idx:, -1]

# tf.keras.utils에서 불러오는 to_categorical은 분류 문제에 자주 쓰이는 함수
# 정답 행렬을 원-핫 인코딩 방식으로 바꿔줌
train_Y = tf.keras.utils.to_categorical(train_Y, num_classes=2)
test_Y = tf.keras.utils.to_categorical(test_Y, num_classes=2)

# 모델 학습
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=48, activation='relu', input_shape=(12,)),
    tf.keras.layers.Dense(units=24, activation='relu'),
    tf.keras.layers.Dense(units=12, activation='relu'),
    # 활성화 함수로 softmax를 사용하면 총합이 1.0인 확률로 변환된다.
    # 큰 값을 강화하고 낮은 값을 약화하는 특성 !!!
    tf.keras.layers.Dense(units=2, activation='softmax')
])
# categorical_crossentropy(CCE)
# loss는 손실 값을 표현하 듯 CCE는 불확실성을 나타낸다. CCE = 0.06 -> Accuracy = 0.94
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.07), loss='categorical_crossentropy',
metrics=['accuracy']) # 분류 문제는 정확도로 퍼포먼스를 측정하기 때문에 정확도 설정은 필수!
model.summary()

history = model.fit(train_X, train_Y, epochs=40, batch_size=32, validation_split=0.25)

# 시각화
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'g-', label='accuracy')
plt.plot(history.history['val_accuracy'], 'k--', label='val_accuracy')
plt.xlabel('Epoch')
plt.ylim(0.7, 1)
plt.legend()

plt.show()

# 모델 성능 평가
model.evaluate(test_X, test_Y)