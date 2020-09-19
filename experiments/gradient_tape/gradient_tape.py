import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import ipdb


inputs = layers.Input(shape=(2,))
action = layers.Dense(3, activation="linear")(inputs)

model = keras.Model(inputs=inputs, outputs=action)

optimizer = keras.optimizers.Adam(learning_rate=20)


target = [1, 4, 5]
target = tf.convert_to_tensor(target)
target = tf.expand_dims(target, 0)
for i in range(1_000):
	# Forward pass
	state = [.1,.2]
	state = tf.convert_to_tensor(state)
	state = tf.expand_dims(state, 0)
	with tf.GradientTape() as tape:
		y = model(state)
		loss = keras.losses.mean_squared_error(target, y)
		print(f"loss {i} {y} {loss}")

	# Calculate gradients with respect to every trainable variable
	# print(f"model.trainable_variables {model.trainable_variables}")
	grads = tape.gradient(loss, model.trainable_variables)
	# print(f"grad {grad}")
	optimizer.apply_gradients(zip(grads, model.trainable_variables))
