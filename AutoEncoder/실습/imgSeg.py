# 이미지 분할(p.378)
# 단순히 이미지의 경계선을 추출하는 작업은 전통적인 알고리즘의 필터나 한 층의 Conv레이어로 처리 가능
# 의미 있는 부분을 추출하고 이미지를 의미있는 부분과 그렇지 않은 부분으로 분할하기 위해서는 학습이 필요
import tensorflow_datasets as tfds
dataset, info = tfds.load('oxford_iiit_pet:3.2.0', with_info=True)
info
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Dataset를 얼마나 반복시켜 학습시킬지를 정해야하기 때문에 따로 저장
train_data_len = info.splits['train'].num_examples
test_data_len = info.splits['test'].num_examples

def load_image(datapoint):
  img = tf.image.resize(datapoint['image'], (128, 128)) # 크기가 크면 tf.keras에서 학습안됨
  mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128)) 

  img = tf.cast(img, tf.float32)
  img = img / 255.0

  mask -= 1

  return img, mask
import tensorflow as tf

train_dataset = dataset['train'].map(load_image)
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.batch(16)

test_dataset = dataset['test'].map(load_image)
test_dataset = test_dataset.repeat()
test_dataset = test_dataset.batch(1)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
for img, mask in train_dataset.take(1):
  plt.figure(figsize=(10, 5))
  
  plt.subplot(1,2,1)
  plt.imshow(img[2])

  plt.subplot(1,2,2)
  plt.imshow(np.squeeze(mask[2], axis=2))
  plt.colorbar()
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def REDNet_segmentation(num_layers):
  conv_layers = []
  deconv_layers = []
  residual_layers = []

  inputs = tf.keras.layers.Input(shape=(None, None, 3))
  conv_layers.append(tf.keras.layers.Conv2D(3, kernel_size=3, padding='same', activation='relu'))

  for i in range(num_layers-1):
    conv_layers.append(tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
    deconv_layers.append(tf.keras.layers.Conv2DTranspose(64, kernel_size=3, padding='same', activation='relu'))
  
  deconv_layers.append(tf.keras.layers.Conv2DTranspose(3, kernel_size=3, padding='same', activation='softmax'))

  # 인코더 시작
  for i in range(num_layers-1):
    x = conv_layers[i+1](x)
    if i % 2 == 0:
      residual_layers.append(x)
  
  # 디코더 시작
  for i in range(num_layers-1):
    if i % 2 == 1:
      x = tf.keras.layers.Add()([x, residual_layers.pop()])
      x = tf.keras.layers.Activation('relu')(x)
    
    x = deconv_layers[i](x)
  
  x = deconv_layers[-1](x)

  model = tf.keras.Model(inputs=inputs, outputs=x)
  return model
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
model = REDNet_segmentation(15)
model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset,
                    epochs=20,
                    steps_per_epoch=train_data_len/16,
                    validation_data=test_dataset,
                    validtaion_steps=test_data_len)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 테스트 이미지 분할 확인
plt.figure(figsize=(12, 12))
for idx, (img, mask) in enumerate(test_dataset.take(3)):
  plt.subplot(3, 3, idx*3 + 1)
  plt.imshow(img[0])

  plt.subplot(3, 3, idx*3+2)
  plt.imshow(np.squeeze(mask[0], axis=2))

  predict = tf.argmax(model.predict(img), axis=-1)
  plt.subplot(3, 3, idx*3+3)
  plt.imshow(np.squeeze(predict[0], axis=0))
