# # 오토인코더 
# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(filters = 32, kernel_size = 2, strides=(2,2), activation = 'elu', input_shape = (64,64,3)),
#     tf.keras.layers.Conv2D(filters = 64, kernel_size = 2, strides=(2,2), activation = 'elu'),
#     tf.keras.layers.Conv2D(filters = 128, kernel_size = 2, strides=(2,2), activation = 'elu'),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation = 'elu'),
#     tf.keras.layers.Dense(64, activation = 'elu'),
#     tf.keras.layers.Dense(8*8*128, activation = 'elu'),
#     tf.keras.layers.Reshape(target_shape=(8,8,128)),
#     tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=2, strides=(2,2), padding ='same', activation = 'elu'),
#     tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=(2,2), padding ='same', activation = 'elu'),
#     tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=2, strides=(2,2), padding ='same', activation = 'sigmoid')
# ])

# model.compile(optimizer =tf.optimizers.Adam(), loss= 'mse', metrics = ['accuracy'])
# model.summary()

# history = model.fit(train_x, train_x, epochs=500, batch_size= 10)

# # 오토인코더 전과 후의 사진
# import random
# import numpy as np

# plt.figure(figsize= (4,8))
# for c in range(4):
#   plt.subplot(4, 2, c*2+1)
#   rand_index = random.randint(0, train_x.shape[0])
#   plt.imshow(train_x[rand_index].reshape(64,64,3), cmap = 'gray')
#   plt.axis('off')

#   plt.subplot(4, 2, c*2+2)
#   img = model.predict(np.expand_dims(train_x[rand_index], axis=0))
#   plt.imshow(img.reshape(64, 64,3), cmap = 'gray')
#   plt.axis('off')

# plt.show()

# model.evaluate(test_x, test_x)

# # 클러스터링 과정
# latent_vector_model = tf.keras.Model(inputs=model.input, outputs=model.layers[5].output)
# #latent 벡터 64개를 추출할 수 있음
# latent_vector = latent_vector_model.predict(train_x)
# print(latent_vector.shape)
# print(latent_vector[0])

# from sklearn.cluster import KMeans 

# kmeans=KMeans(n_clusters=5, n_init=10)
# kmeans.fit(latent_vector)

# plt.figure(figsize=(12, 12))

# for i in range(5):
#   images = train_x[kmeans.labels_==i]
#   for c in range(10):
#     plt.subplot(10, 10, i*10+c+1)
#     plt.imshow(images[c].reshape(64,64, 3))
#     plt.axis('off')

# plt.show()

# # 클러스터링을 TSNE으로 시각화

# %time
# from sklearn.manifold import TSNE

# tsne = TSNE(n_components = 2, learning_rate = 150, perplexity = 15)
# print(latent_vector[:300].shape)
# tsne_vector = tsne.fit_transform(latent_vector[:300])

# cmap = plt.get_cmap('rainbow', 5)#10
# fig = plt.scatter(tsne_vector[:,0], tsne_vector[:,1], marker = '.', c=train_y[:300], cmap = cmap)
# cb = plt.colorbar(fig, ticks=range(5))
# n_clusters = 5
# tick_locs = (np.arange(n_clusters)+ 0.5)*(n_clusters-1)/n_clusters
# cb.set_ticks(tick_locs)
# cb.set_ticklabels(range(10))

# plt.show()

# %%time

# perplexities = [5, 10, 15, 25, 50, 100]
# plt.figure(figsize = (8,12))

# for c in range(6):
#     tsne = TSNE(n_components = 2, learning_rate = 100, perplexity = perplexities[c])
#     tsne_vector = tsne.fit_transform(latent_vector[:5000])
    
#     plt.subplot(3, 2, c+1)
#     plt.scatter(tsne_vector[:,0], tsne_vector[:,1], marker='.', c=train_y[:5000], cmap = 'rainbow')
#     plt.title('perplexity:{0}'.format(perplexities[c]))