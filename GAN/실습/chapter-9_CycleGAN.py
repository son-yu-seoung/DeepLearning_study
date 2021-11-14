from __future__ import print_function, division
import scipy
from tensorflow.keras.datasets import mnist
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import DataLoader

 

class CycleGAN:
    def __init__(self):
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.dataset_name = 'apple2orange' # 데이터 로더 설정
        self.data_loader = DataLoader.DataLoader(dataset_name=self.dataset_name, img_res=(self.img_rows, self.img_cols))

         # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4) # D(PatchGAN)의 출력 크기를 계산 
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64

        # Loss weights
        self.lambda_cycle = 10.0 # 사이클-일관성 손실 가중치, 높으면 원본 이미지와 재구성 이미지가 가능한 아주 비슷하게 만들어진다., CycleGAN 저자들은 이 값들이(특히 훈련 과정 초기) 변화에 얼마나 극적으로 영향을미치즌지 이야기한다.
        self.lambda_id = 0.9 * self.lambda_cycle # 동일성 손실 가중치, 이 값이 낮으면 불필요한 변화가 생긴다 예를 들어 초기에 색이 완전히 반전된다.

        optimizer = Adam(0.0002, 0.5)
        
        # Build and compile the discriminators 
        self.d_A = self.build_discriminator() # 판별자 두 개 생성
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss='mse', # 손실 함수가 mse인 것 주목!
                         optimizer=optimizer,
                         metrics=['accuracy'])
        self.d_B.compile(loss='mse',
                         optimizer=optimizer,
                         metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Build the generators
        self.g_AB = self.build_generator() # 생성자 두 개 생성
        self.g_BA = self.build_generator()

        # Input images from both domains
        img_A = Input(shape=self.img_shape) # 이미지를 넣은 것이 아니라 현재는 shape만 있음
        img_B = Input(shape=self.img_shape) # 두 도메인의 입력 이미지 

        # Translate images to the other domain
        fake_B = self.g_AB(img_A) # 생성된 B
        fake_A = self.g_BA(img_B) # 생성된 A
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B) # 원본 도메인으로 이미지를 다시 변환한다.
        reconstr_B = self.g_AB(fake_A)
        # Identity mapping of images
        img_A_id = self.g_BA(img_A) # 동일한 이미지 매핑
        img_B_id = self.g_AB(img_B)

        # For the combined model we will only train the generators
        self.d_A.trainable = False # 연결 모델에서는 생성자만 훈련한다.
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A) # 판별자가 변환된 이미지의 유효성을 결정한다.
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[valid_A, valid_B,
                                       reconstr_A, reconstr_B,
                                       img_A_id, img_B_id])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                              loss_weights=[1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id],
                              optimizer=optimizer)

test = CycleGAN()
