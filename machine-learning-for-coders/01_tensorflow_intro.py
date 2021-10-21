import tensorflow as tf
import numpy as np
import tensorflow.keras as k
import tensorflow.keras.layers as kl

layer0 = kl.Dense(units=1, input_shape=[1])
model = k.Sequential([layer0])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0])

model.fit(xs,ys, epochs=500)

print(model.predict([10.0]))
print(layer0.get_weights())
# =>
# [[18.986605]]
# [array([[1.9980586]], dtype=float32), array([-0.9939808], dtype=float32)]


