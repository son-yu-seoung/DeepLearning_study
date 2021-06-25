from tensorflow import keras
import dataProcessing as dp
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
# from sklearn.model_selection import train_test_split
from PIL import Image

globalPath = 'C:\\2021-son\\AnimalProject\\dataset\\'
ds = dp.dataSet(globalPath)
train_X, train_Y, test_X, test_Y = ds.load_data(64, 0.9)

# print("train_data Load")
# train_X = np.load("C:\\2021-son\\AnimalProject\\dataset\\Numpy\\train_X.npy")
# train_Y = np.load("C:\\2021-son\\AnimalProject\\dataset\\Numpy\\train_Y.npy")

# print("test_data Load")
# test_X = np.load("C:\\2021-son\\AnimalProject\\dataset\\Numpy\\test_X.npy")
# test_Y = np.load("C:\\2021-son\\AnimalProject\\dataset\\Numpy\\test_Y.npy")



# print(train_X.shape, test_X.shape) 
# print(train_Y.shape, test_Y.shape)
# print(test_Y[:10])
# img = train_X[100] 
# img = Image.fromarray(np.uint8(img))# 이미지 출력
# img.show() #이미지 출력

model = tf.keras.Sequential([
    # tf.keras.layers.Conv2D(input_shape=(64, 64, 3), kernel_size=(3,3), filters=6, padding='same'),
    # tf.keras.layers.Conv2D(kernel_size=(3,3), filters=12, padding='same'),
    # tf.keras.layers.Conv2D(kernel_size=(3,3), filters=32, padding='same'),    
    # tf.keras.layers.Flatten(),
    # tf.keras.layers.Dense(units=128, activation='relu'),
    # tf.keras.layers.Dense(units=5, activation='softmax')#accuracy는 91%나오는데 val_accuracy는 5~60% ??
    tf.keras.layers.Conv2D(input_shape=(64, 64, 3), kernel_size=(3,3), filters=32, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(kernel_size=(3,3), filters=64, padding='same',activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    # tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.Conv2D(kernel_size=(3,3), filters=64, padding='same',activation='relu'),
    tf.keras.layers.Conv2D(kernel_size=(3,3), filters=128, padding='same',activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.Flatten(), # 잠재 변수 
    tf.keras.layers.Dense(units=512, activation='relu'),
    # tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.Dense(units=5, activation='softmax') #units = 10 해야하나?
])

model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.summary()

history = model.fit(train_X, train_Y, epochs=20, validation_split=0.25) #vaildation = 0.25

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'g-', label='accuracy')
plt.plot(history.history['val_accuracy'], 'k--', label='val_accuracy')
plt.xlabel('Epochs')
plt.ylim(0.7, 1)
plt.legend()

plt.show()

print(model.evaluate(test_X, test_Y, verbose=0))
