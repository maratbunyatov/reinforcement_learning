import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.losses as kls
import cv2
import ipdb
import sys


EPISODES = 2000
gamma = 0.99  # Discount factor for past rewards
LEARNING_RATE = 0.01
env = gym.make('CartPole-v1')


inputs = layers.Input(shape=(env.observation_space.shape[0] + env.action_space.n,))
common = layers.Dense(128, activation="relu")(inputs)
critic = layers.Dense(1)(common)
model = keras.Model(inputs=inputs, outputs=critic)
model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'])

def returns(rewards, dones):
    # `next_value` is the bootstrap value estimate of the future state (critic).
    returns = np.append(np.zeros_like(rewards), [0], axis=-1)
    # Returns are calculated as discounted sum of future rewards.
    # ipdb.set_trace()
    for t in reversed(range(rewards.shape[0])):
        returns[t] = rewards[t] + gamma * returns[t + 1] * (1 - dones[t])
    returns = returns[:-1]
    return returns

def test():
	batch_size = 10
	for i in range(EPISODES):
		# batch_size = np.random.randint(1,10)
		Y = [[5.]] * batch_size 
		# Y = Y.reshape(-1,1)
		Y = tf.expand_dims(Y, 0)
		x = [[5., 4.]] * batch_size
		x = tf.expand_dims(x, 0)

		# ipdb.set_trace()
		model.train_on_batch(x, Y)

	print(model.predict_on_batch(x))
	sys.exit()
# test()


x = None
for episode in range(EPISODES):
	done = False
	state = env.reset()
	ep_rewards = 0
	states = []
	actions = []
	rewards = []
	dones = []
	values = []
	while not done:
		env.render()           
		left = np.array([1., 0.])
		right = np.array([0., 1.])
		# model.predict_on_batch
		activations = model.predict_on_batch(np.vstack((np.concatenate((state, left)), np.concatenate((state, right)))))
		probs = tf.nn.softmax(tf.squeeze(np.array(activations)))
		action = tf.squeeze(tf.random.categorical(tf.expand_dims(probs, 0), 1)).numpy()
		# action = np.argmax(probs)
		state_, reward, done, _ = env.step(action)
		if done:
			reward = 0

		states.append(state)
		actions.append(tf.one_hot(indices=[action], depth=2))
		rewards.append(reward)
		dones.append(done)
		# values.append([left, right][action])

		ep_rewards += reward * (1-int(done))
		if not done:
			state = state_

	print(f"episode: {episode}, ep_rewards: {ep_rewards}")

	states = np.array(states)
	actions = np.array(actions)
	rewards = np.array(rewards)
	dones = np.array(dones)
	Y = returns(rewards, dones).reshape(-1,1) #- np.squeeze(values)
	Y = tf.expand_dims(Y, 0)
	x = np.hstack((states, np.array(actions.reshape(-1,2))))
	x = tf.expand_dims(x, 0)
	# ipdb.set_trace()
	model.train_on_batch(x, Y)

ipdb.set_trace()

