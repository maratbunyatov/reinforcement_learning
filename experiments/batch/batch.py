import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import ipdb


inputs = layers.Input(shape=(2,))
common = layers.Dense(128, activation="relu")(inputs)
critic = layers.Dense(3)(common)
model = keras.Model(inputs=inputs, outputs=critic)
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])

batch_size = 1_000

Y = [[1, 4, 5]] * batch_size 
# Y = [1, 4, 5]
# Y = tf.convert_to_tensor(Y)
Y = tf.expand_dims(Y, 0)

x = [[5, 4]] * batch_size
# x = tf.convert_to_tensor(x)
x = tf.expand_dims(x, 0)

for i in range(200):
	# Forward pass
	# model.fit(tf.expand_dims(x, 0), tf.expand_dims(Y, 0), batch_size=batch_size)
	model.train_on_batch(x, Y)

# ipdb.set_trace()
print(model.predict(tf.expand_dims(x[0, 0], 0)))