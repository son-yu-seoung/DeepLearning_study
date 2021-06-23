# 다항 분류의 한 종류인 MNIST 데이터
# tf.keras에 데이터셋이 있기 때문에 편하게 불러올 수 있다.
import tensorflow as tf
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()

print(len(train_X), len(test_X)) # 6만 : 1만

# 이미지 확인 해보기
plt.imshow(train_X[0], cmap='gray') # cmap이 뭘까? colormap의 약어
plt.colorbar() # 채도 
plt.show()

# 정규화 0 ~ 1까지 
train_X = train_X / 255.0 
test_X = test_X / 255.0

# 해당 코드에서 to_categorical를 사용하게 되면 [0,0,0,0,0,0,0,1,0,0]과 같이 0이라는 데이터를 주는 비효율적인 작업을 하게 됩니다.
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] 대신 9로 표시하게 되면 데이터의 낭비가 없어짐
# 이미 train_Y, test_Y는 이런식으로(9) 표현 되어있다.
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=10, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy' # 해당 loss방법은 별도의 데이터 전처리 없이 희소 행렬을 나타내는 데이터를 정답 행렬로 사용
             , metrics=['accuracy'])
model.summary() 

history = model.fit(train_X, train_Y, epochs=25, validation_split=0.25)

# 학습 결과를 시각화
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'b-', label='accuracy')
plt.plot(history.history['val_accuracy'], 'r--', label='val_accuracy')
plt.xlabel("Epoch")
plt.ylim(0.7, 1)
plt.legend()

plt.show()

# 성능 평가, 정확도는 88%정도가 나온다. CNN으로 더 정확률을 올릴 수 있다.
model.evaluate(test_X, test_Y)