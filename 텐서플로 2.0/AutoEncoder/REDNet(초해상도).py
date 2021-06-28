# colab환경에서 코딩

import tensorflow as tf
# BSD 데이터세트 불러오기
tf.keras.utils.get_file('/content/bsd_images.zip', 'http://bit.ly/35pHZlC', extract=True)
!unzip /content/bsd_images.zip

import pathlib, glob
image_root = pathlib.Path('/content/images')

all_image_paths = list(image_root.glob('*/*'))
print(all_image_paths[:10])
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import PIL.Image as Image
import matplotlib.pyplot as plt

plt.figure(figsize=(12,12))
for c in range(9):
  plt.subplot(3, 3, c+1)
  plt.imshow(plt.imread(all_image_paths[c]))
  plt.title(all_image_paths[c])
  plt.axis('off')
plt.show()
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# BSD500은 200장의 훈련 데이터, 100장의 검증 데이터, 200장의 테스트 데이터로 구성되어있음
# 각 데이터세트 집합을 처리하기 위한 이미지의 경로 분리
train_path, valid_path, test_path = [], [], []

for image_path in all_image_paths:
  if str(image_path).split('.')[-1] != 'jpg':
    continue
  if str(image_path).split('/')[-2] == 'train':
    train_path.append(str(image_path))
  elif str(image_path).split('/')[-2] == 'val': # == 'valid'로 해놔서 계속 오류 났던 것 항상 조심하고 고쳐지지 않는 문제가 생겼을 때는 파일 처음부터 되돌아보기!
                                                # 오류가 나는 이유 -> 내 코드에 오류가 있다는 것 -> 절대 못 찾을게 아니다 침착하게 순서대로 짚어보자!
    valid_path.append(str(image_path))
  else:
    test_path.append(str(image_path))
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 초해상도 얻는 방법 
# 1. 원본이미지에서 crop으로 고해상도를 얻어옴
# 2. 고해상도를 1/2 축소하고 2배 확대를하여 저해상도를 얻음
# 3. 입력으로 저해상도를 넣고 RedNet을 통과하면 출력으로 고해상도를 다시 얻음

# 원본이미지에서 고해상도와 저해상도를 만드는 함수 
def get_hr_and_lr(image_path):
  img = tf.io.read_file(image_path)
  img = tf.image.decode_jpeg(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32) # 딥러닝에서 float32를 가장 보편적으로 사용

  hr = tf.image.random_crop(img, [50, 50, 3])
  lr = tf.image.resize(hr, [25, 25]) 
  lr = tf.image.resize(lr, [50, 50]) # input
  return lr, hr

# dataset 정의 
train_dataset = tf.data.Dataset.list_files(train_path)
train_dataset = train_dataset.map(get_hr_and_lr)
train_dataset = train_dataset.repeat() # 데이터를 반복적으로 사용하기위한 repeat, batch
train_dataset = train_dataset.batch(16) # 이거 왜 하는지?

valid_dataset = tf.data.Dataset.list_files(valid_path)
valid_dataset = valid_dataset.map(get_hr_and_lr)
valid_dataset = valid_dataset.repeat()
valid_dataset = valid_dataset.batch(1)

# REDNet-30을 정의, 모델 구현
def REDNet(num_layers): # num_layers는 컨볼루션 레이어와 디컨볼루션 레이어의 수 REDNet-30이라면 15를 넣으면 됨
  # 레이어가 많고 서로 연산을 해야 하기 때문에 각 레이어를 저장할 리스트가 따로 필요
  conv_layers = []
  deconv_layers = []
  residual_layers = [] #

  inputs = tf.keras.layers.Input(shape=(None, None, 3))
  conv_layers.append(tf.keras.layers.Conv2D(3, kernel_size=3, padding='same', activation='relu')) # filter = 3 (채널 받기 위해)

  for i in range(num_layers-1):
    conv_layers.append(tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
    deconv_layers.append(tf.keras.layers.Conv2DTranspose(64, kernel_size=3, padding='same'))
  
  deconv_layers.append(tf.keras.layers.Conv2DTranspose(3, kernel_size=3, padding='same')) # filter = 3 (채널 받기 위해)

  x = conv_layers[0](inputs)

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
# 모델을 컴파일, 고해상도 이미지가 잘 복원됐는지 알기 위해서 특별한 측정값을 추가
# PSNR(Peak signal-toNoise Ratio) : '신호 대 잡음비' 
def psnr_metric(y_true, y_pred):
  return tf.image.psnr(y_true, y_pred, max_val=1.0)

model = REDNet(15)
model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss='mse', metrics=[psnr_metric])
# # 네트워크 시각화(!중요!, 가끔씩 유용하게 사용될 수 있을 것 같음)
# tf.keras.utils.plot_model(model) 

# Dataset을 이용한 학습은 fit대신 fit_generator
history = model.fit_generator(train_dataset,
                              epochs=1000,
                              steps_per_epoch=len(train_path)//16,
                              validation_data=valid_dataset,
                              validation_steps=len(valid_path),
                              verbose=2)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
plt.plot(history.history['psnr_metric'], 'b-', label='psnr')
plt.plot(history.history['val_psnr_metric'], 'r--', label='val_psnr')
plt.xlabel('Epoch')
plt.legend()
plt.show()
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
img = tf.io.read_file(test_path[0])
img = tf.image.decode_jpeg(img, channels=3)
hr = tf.image.convert_image_dtype(img, tf.float32)

lr = tf.image.resize(hr, [hr.shape[0]//2, hr.shape[1]//2])
lr = tf.image.resize(lr, [hr.shape[0], hr.shape[1]])
predict_hr = model.predict(np.expand_dims(lr, axis=0))

print(tf.image.psnr(np.squeeze(predict_hr, axis=0), hr, max_val=1.0))
print(tf.image.psnr(lr, hr, max_val=1.0))
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
plt.figure(figsize=(16,4))

plt.subplot(1, 3, 1)
plt.imshow(hr)
plt.title('original - hr')

plt.subplot(1, 3, 2)
plt.imshow(lr)
plt.title('lr')

plt.subplot(1, 3, 3)
plt.imshow(np.squeeze(predict_hr, axis=0))

plt.show()
