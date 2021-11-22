import numpy as np
import tensorflow as tf

x = []
y = [] 

for i in range(6):
    lst = list(range(i, i+4))

    x.append(list(map(lambda c: [c/10], lst)))
    y.append(((i+4)/10))

X = np.array(x) # Numpy version error -> Numpy down grade 1.19.5
Y = np.array(y)

for i in range(len(Y)):
    print(f'{X[i]}, {Y[i]}')

model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(units=10, return_sequences=False, input_shape=[4, 1]),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

model.fit(X, Y, epochs=1000)
print(model.predict(X))

print(model.predict(np.array([[[0.6], [0.7], [0.8], [0.9]]])))
print(model.predict(np.array([[[-0.1], [0.0], [0.1], [0.2]]])))