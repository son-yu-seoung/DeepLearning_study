# 오토인코더는 입력에 가까운 인코더(Encoder), 잠재 변수(Latent Vector), 출력에 가까운(Decoder)
# 인코더는 입력에서 잠재 변수를 만드는 부분, 디코더는 잠재 변수를 출력으로 만드는 부분
# 인코더는 특징 추출기 역할 : 입력 이미지에서 특징을 추출하여 1차원의 벡터(잠재 변수)로 만듬

# MNIST 데이터로 실습
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(train_X, train_Y), (test_X, test_Y) = mnist.load_data()
print(train_X.shape, train_Y.shape) # (60000, 28, 28) (60000,)
 
train_X = train_X / 255.0
test_X = test_X / 255.0

# 이미지 확인
plt.imshow(train_X[0], cmap='gray')
plt.colorbar()
#plt.show()
print(train_Y[0])

# 모델 학습
train_X = train_X.reshape(-1, 28, 28, 1)
test_X = test_X.reshape(-1, 28, 28, 1)

# 활성화 함수로 relu를 사용하면 복원된 이미지가 각지고 손실이 많이 일어난다.
# 이유로는 relu는 0이하의 값이 들어오면 0으로 고정시켜버리기 때문
# 그런 이유로 elu를 사용한다 elu는 0이하의 값이 들어오면 서서히 -1에 수렴시키는 함수(값이 relu보다는 보정됨)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=2, strides=(2,2), activation='elu', input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=2, strides=(2,2), activation='elu'),
    tf.keras.layers.Flatten(), # 3차원 데이터를 1차원으로 바꿔주기 위해, 1
    tf.keras.layers.Dense(units=64, activation='elu'), # 잠재 변수, 2    ----> 1, 2는 한 몸
    tf.keras.layers.Dense(units=7*7*64, activation='elu'), # 1 Flatten과 대칭됨 !!!!
    tf.keras.layers.Reshape(target_shape=(7, 7, 64)), # 2    -----> 1, 2는 한 몸
    tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=(2,2), padding='same', activation='elu'),
    tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=2, strides=(2,2), padding='same', activation='elu')
])

model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse', metrics=['accuracy'])
model.summary()

model.fit(train_X, train_X, epochs=20, batch_size=256)

# 테스트 데이터로 컨볼루션 오토인코더의 이미지 재생성
import random
import numpy as np

plt.figure(figsize=(4,8))
for c in range(4):
    plt.subplot(4, 2, c*2+1)
    rand_index = random.randint(0, test_X.shape[0])
    plt.imshow(test_X[rand_index].reshape(28, 28), cmap='gray')
    plt.axis('off')

    plt.subplot(4, 2, c*2+2)
    img = model.predict(np.expand_dims(test_X[rand_index], axis=0))
    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.show()

model.evaluate(test_X, test_X) # 정확도가 왜 이렇게 낮은지 확인해보기 

