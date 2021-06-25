# 오토인코더는 입력에 가까운 인코더(Encoder), 잠재 변수(Latent Vector), 출력에 가까운(Decoder)
# 인코더는 입력에서 잠재 변수를 만드는 부분, 디코더는 잠재 변수를 출력으로 만드는 부분
# 인코더는 특징 추출기 역할 : 입력 이미지에서 특징을 추출하여 1차원의 벡터(잠재 변수)로 만듬

# MNIST 데이터로 실습
from re import A
from threading import Timer
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

model.fit(train_X, train_X, epochs=1, batch_size=256)

# 테스트 데이터로 컨볼루션 오토인코더의 이미지 재생성
import random
import numpy as np

plt.figure(figsize=(4,8))
for c in range(1):
    plt.subplot(4, 2, c*2+1)
    rand_index = random.randint(0, test_X.shape[0])
    plt.imshow(test_X[rand_index].reshape(28, 28), cmap='gray')
    plt.axis('off')

    plt.subplot(4, 2, c*2+2)
    #img = model.predict(test_X[rand_index])
    img = model.predict(np.expand_dims(test_X[rand_index], axis=0)) # axis=0이 하는 역할? : (28, 28, 1) -> (1, 28, 28, 1)로 바뀜 !
    print(img.shape)
    print(test_X[0].shape)
    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.show()
print(img.shape)
print(test_X[0].shape)

model.evaluate(test_X, test_X) 

# 클러스터링(Clurstering)
# 잠재변수를 이용해 데이터를 여러 개의 군집으로 클러스터링 할 수 있다.(비지도학습)
# K-평균 클러스터링
latent_vector_model = tf.keras.Model(inputs=model.input, outputs=model.layers[3].output)
latent_vector = latent_vector_model.predict(train_X)
print(latent_vector.shape) # (60000,64)
print(latent_vector[0]) 

from sklearn.cluster import KMeans
# n_clusters = 클러스터 중심의 개수, n_init = 알고리즘 실행 횟수, random_state = 알고리즘의 계산 결과를 동일하게 가져가기 위해 지정하는 랜덤 초기화 숫자
kmeans = KMeans(n_clusters=10, n_init=10, random_state=42)
kmeans.fit(latent_vector)

print(kmeans.labels_) # 0~9사이 중 어떤 클러스터에 속하는지! (숫자 0~9가 아니라 ~번째 클러스터)
print(kmeans.cluster_centers_.shape) 
print(kmeans.cluster_centers_[0])

# 각 클러스터에 속하는 이미지가 어떤 이미지인지 출력해서 확인해보기
plt.figure(figsize=(12,12))

for i in range(10): # 0~9
    images = train_X[kmeans.labels_ == i] # 
    for c in range(10):
        plt.subplot(10, 10, i*10+c+1)
        plt.imshow(images[c].reshape(28,28), cmap='gray')
        plt.axis('off')
plt.show()

# 잠재변수의 차원 수를 늘리거나 KMeans()의 n_init을 늘려서 좀 더 분류가 잘 되도록 시도해볼 수 있다.
# 또는 n_cluster를 늘려서 클러스터를 
# 더욱 세분화할 수 있다.


# 하지만 클러스터를 시각화 하는 문제가 남아있다., 클러스터링은 잠재변수에서 하는 것이기에 잠재변수는 64차원
# 클러스터링 결과를 시각화할 수는 없을까? 2차원 혹은 3차원으로 잠재변수가 가진 차원을 축소해야함
# t-SNE은 여기에 최적화된 알고리즘 p.349
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, learning_rate=100, perplexity=15, random_state=0)
# n_componets=차원수, learning_rate=10~1000, perplexity=알고리즘 계산에서 고려할 최근접 이웃의 숫자,
# perplexity를 몇으로 설정하냐에 따라 데이터 모양이 달라지므로 여러번 실험해봐야함
tsne_vector = tsne.fit_transform(latent_vector[:5000])
# 학습과 변환 과정을 동시에 진행하는 함수, 함수로 결과값을 반환
#  훈련 데이터 중 5,000개만 사용(많아지면 계산 속도가 느려짐 64차원이라)

cmap = plt.get_cmap('rainbow', 10)
fig = plt.scatter(tsne_vector[:, 0], tsne_vector[:, 1], marker='.', c=train_Y[:5000], cmap=cmap)
cb = plt.colorbar(fig, ticks=range(10))
n_clusters = 10
tick_locs = (np.arange(n_clusters) + 0.5) * (n_clusters-1) / n_clusters

cb.set_ticks(tick_locs)
cb.set_ticklabels(range(10))

plt.show()

# 클러스터 분리 결과를 좀 더 직관적으로 확인하기 위해서 t-SNE로 분리된 클러스터위에 MNIST출력 !!!!
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox

plt.figure(figsize=(16, 16))

tsne = TSNE(n_components=2, learning_rate=100, perplexity=15, random_state=0)
tsne_vector = tsne.fit_transform(latent_vector[:5000])

ax = plt.subplot(1, 1, 1)
ax.scatter(tsne_vector[:, 0], tsne_vector[:, 1], marker='.', c=train_Y[:5000], cmap='rainbow')
for i in range(200):
    imagebox = OffsetImage(train_X[i].reshape(28, 28))
    # AnnotationBbox : 이미지나 텍스트 등의 주석을 그래프 위에 표시하기 위한 함수 
    ab = AnnotationBbox(imagebox, (tsne_vector[i,0], tsne_vector[i,1]), frameon=False, pad=0.0)
    ax.add_artist(ab)
ax.set_xticks([])
ax.set_yticks([])
plt.show()