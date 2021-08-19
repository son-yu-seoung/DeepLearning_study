import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout, Flatten, Reshape
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)
z_dim = 100 

def build_generator(z_dim):

    model = Sequential()

    model.add(Dense(256*7*7, input_dim=z_dim)) # input_dim = 1개면 1차원 n개면 n개의 노드가 입력으로 들어간다고 생각하면 된다. 
    model.add(Reshape((7, 7, 256)))

    model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same')) # 14 14
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same')) # 14 14
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    
    model.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same')) # 28 28
    model.add(Activation('tanh'))

    return model 

def build_discriminator(img_shape):
    
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding='same')) # 14 14
    model.add(LeakyReLU(alpha=0.01))

    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same')) # 7 7 
    model.add(LeakyReLU(alpha=0.01))

    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same')) # 3 3
    model.add(LeakyReLU(alpha=0.01))

    model.add(Flatten()) # 3 * 3 * 128
    model.add(Dense(1, activation='sigmoid'))

    return model 

def build_gan(generator, discriminator):

    model = Sequential()

    model.add(generator) # 생성자 판별자 모델 연결 
    model.add(discriminator)

    return model

discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])
discriminator.trainable = False

generator = build_generator(z_dim)

gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=Adam())

losses = [] # 그래프를 그리기 위해서
accuracies = [] # 일정 간격마다 append 
iteration_checkpoints = []

def train(iterations, batch_size, sample_interval):

    (train_x, _), (_, _) = mnist.load_data()

    train_x = train_x / 127.5 - 1.0 # [0, 255] 흑백 픽셀 값을 [-1, 1]로 스케일 조정
    train_x = np.expand_dims(train_x, axis=3) # (60000, 28, 28) -> (60000, 28, 28, 1)

    real = np.ones((batch_size, 1)) # shape = (batch_size, 1), not 1 dim

    fake = np.zeros((batch_size, 1))

    for iteration in range(iterations): # Epoch라고 생각하면 될 듯 
        print("{}번째 Epoch".format(iteration))

        idx = np.random.randint(0, train_x.shape[0], batch_size) # 0부터 train_x[0]의 개수까지 batch_size만큼 랜덤 idx 생성, shape = (batch_size ,)
        imgs = train_x[idx] # train_x[[1, 77, 24, 24 , ...]], 해당 idx의 train data를 모두 반환

        z = np.random.normal(0, 1, (batch_size, 100)) # noise 생성 0부터 1까지의 정규 분포 생성 shape = (batch_size, 100)
        gen_imgs = generator.predict(z) # 학습단계가 아닌 예측을 내보내는 단계기 떄문에 predict

        # discriminator.trainable = True # trainable test 
        d_loss_real = discriminator.train_on_batch(imgs, real) # 판별자 훈련, trainable = False로 default값이 되어있는데 괜찮은건가 
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake) # (x, y)
        d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)
        # discriminator.trainable = False 

        z = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(z)

        g_loss = gan.train_on_batch(z, real) # 생성자 훈련, 판별자를 속인것으로 만들어서 generator가 학습할 수 있도록 

        if (iteration + 1 ) % sample_interval == 0:

            losses.append((d_loss, g_loss)) 
            accuracies.append(100.0 * accuracy)
            iteration_checkpoints.append(iteration + 1)

            print("{} [ D 손실: {}, 정확도: {}%] [ G 손실: {}".format(iteration + 1, d_loss, 100.0 * accuracy, g_loss))

            sample_images(generator)

def sample_images(generator, image_grid_rows=4, image_grid_columns=4):

    z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))

    gen_imgs = generator.predict(z)
    gen_imgs = 0.5 * gen_imgs + 0.5 # 이미지 픽셀 값을 [0, 1] 범위로 스케일 조정...

    fig, axs = plt.subplots(image_grid_rows, image_grid_columns, figsize=(4,4), sharey=True, sharex=True)
    cnt = 0
    
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    plt.show()

train(iterations=1000, batch_size=128, sample_interval=200)



    