# 이미지 분류를 위한 컨볼루션 신경망은 특징 추출기와 분류기가 합쳐져 있는 형태
# 컨볼루션 신경망을 구성하는 레이어 : 컨볼루션 레이어, 풀링 레이어, 드랍아웃 레이어
import tensorflow as tf
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()

# 정규화 0 ~ 1까지 
train_X = train_X / 255.0 
test_X = test_X / 255.0

# Conv2D 레이어는 채널을 가진 형태의 데이터를 받고록 기본적으로 설정되어 있기 때문에 채널을 갖도록 reshape필요
print(train_X.shape, test_X.shape) # (60000, 28, 28) (10000, 28, 28)

train_X = train_X.reshape(-1, 28, 28, 1)
test_X = test_X.reshape(-1, 28, 28, 1)

print(train_X.shape, test_X.shape) # (60000, 28, 28, 1) (10000, 28, 28, 1)

# # 데이터 확인하기
# plt.figure(figsize=(10, 10))
# for c in range(16):
#     # 4행 4열로 지정한 그리드에서 c+1번째의 칸에 그래프를 그린다.
#     plt.subplot(4, 4, c+1)
#     plt.imshow(train_X[c].reshape(28, 28), cmap='gray')

# plt.show()

# # CNN 모델 구현
# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(input_shape=(28, 28, 1), kernel_size=(3,3), filters=32),
#     tf.keras.layers.MaxPool2D(strides=(2,2)),
#     tf.keras.layers.Conv2D(kernel_size=(3,3), filters=64),
#     tf.keras.layers.MaxPool2D(strides=(2,2)),
#     tf.keras.layers.Conv2D(kernel_size=(3,3), filters=128),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(units=128, activation='relu'),
#     tf.keras.layers.Dropout(rate=0.3),
#     tf.keras.layers.Dense(units=10, activation='softmax')
# ])

# CNN(VGGNet) 구현
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(input_shape=(28, 28, 1), kernel_size=(3,3), filters=32, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(kernel_size=(3,3), filters=64, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(strides=(2,2)),
    tf.keras.layers.Conv2D(kernel_size=(3,3), filters=128, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(kernel_size=(3,3), filters=256, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(strides=(2,2)),
    tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.Dense(units=10, activation='softmax'),
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
 loss='sparse_categorical_crossentropy',
 metrics=['accuracy'])
model.summary()

history = model.fit(train_X, train_Y, epochs=25, validation_split=0.25)

# 학습 결과 그래프로 보기
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
plt.legend()
plt.ylim(0.7, 1)

plt.show()

# 성능 평가
model.evaluate(test_X, test_Y, verbose=0)

# 테스트 데이터에 대한 분류의 성적은 89% 정도로 CNN사용 전보다는 올랐지만 아직 부족하다.
# CNN 퍼포먼스 올리는 법 (1. 더 많은 레이어 쌓기, 2. 이미지 보강(Image Augmentation))
# 1. 점점 더 깊어지는 신경망 중에서 대표적인 레이어들은
# LeNet -> AlexNet -> VGGNet -> GoogLeNet -> ResNet 등이 있다.
# 2. 이미지 보강이란 훈련 데이터에 없는 이미지를 새롭게 만들어내서 훈련 데이터를 보강하는 것 p.166



