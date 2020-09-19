import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.losses as kls
import cv2
import ipdb
import sys


EPISODES = 20_000
gamma = 0.99  # Discount factor for past rewards
LEARNING_RATE = 0.01
env = gym.make('CartPole-v1')


inputs = layers.Input(shape=(env.observation_space.shape[0],))
common = layers.Dense(128, activation="relu")(inputs)
output = layers.Dense(env.action_space.n)(common)
model = keras.Model(inputs=inputs, outputs=output)
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

def getY(returns, actions, activations):
    result = []
    # ipdb.set_trace()
    for i, a in enumerate(actions):
        if not a:
            result.append([returns[i][0], activations[i][1]])
        else:
            result.append([activations[i][0], returns[i][0]])
    return np.array(result)

def test():
	batch_size = 1
	for i in range(EPISODES):
		# batch_size = np.random.randint(1,10)
		Y = [[ 2.45280713e+01,  4.84942272e-03],
	       [ 2.37657286e+01,  1.30628813e-02],
	       [ 1.12816274e-01,  2.29956854e+01],
	       [ 5.24941385e-02,  2.22178641e+01],
	       [-5.07449731e-03,  2.14321859e+01],
	       [ 2.06385716e+01,  4.37170491e-02],
	       [ 1.98369410e+01,  8.60459264e-03],
	       [ 1.90272132e+01,  9.57731716e-03],
	       [ 1.03900708e-01,  1.82093062e+01],
	       [ 1.73831376e+01,  8.67705885e-03],
	       [ 9.98666659e-02,  1.65486239e+01],
	       [ 3.93227190e-02,  1.57056807e+01],
	       [-3.28526599e-03,  1.48542229e+01],
	       [ 1.39941645e+01,  4.59023081e-02],
	       [-5.95266465e-03,  1.31254187e+01],
	       [-1.36202937e-02,  1.22478977e+01],
	       [ 1.13615128e+01,  8.31669420e-02],
	       [ 1.04661746e+01,  4.69657704e-02],
	       [-1.35897826e-02,  9.56179250e+00],
	       [ 8.64827525e+00,  4.85857576e-02],
	       [-1.67772006e-02,  7.72553056e+00],
	       [-2.53011789e-02,  6.79346521e+00],
	       [ 5.85198506e+00,  8.85631144e-02],
	       [-3.21348235e-02,  4.90099501e+00],
	       [-4.18403000e-02,  3.94039900e+00],
	       [ 2.97010000e+00,  1.30572036e-01],
	       [-5.23412377e-02,  1.99000000e+00],
	       [-6.42506778e-02,  1.00000000e+00],
	       [ 0.00000000e+00,  1.75459012e-01]] * batch_size 
		# Y = Y.reshape(-1,1)
		Y = tf.convert_to_tensor(Y)[-12:]
		# Y = tf.expand_dims(Y, 0)
		x = [[ 3.53783063e-02,  1.09937534e-02, -4.10101012e-02,
	        -2.78247199e-02],
	       [ 3.55981813e-02, -1.83516820e-01, -4.15665956e-02,
	         2.51642234e-01],
	       [ 3.19278449e-02, -3.78021323e-01, -3.65337510e-02,
	         5.30929919e-01],
	       [ 2.43674185e-02, -1.82405031e-01, -2.59151526e-02,
	         2.26962745e-01],
	       [ 2.07193178e-02,  1.30775083e-02, -2.13758977e-02,
	        -7.37808085e-02],
	       [ 2.09808680e-02,  2.08499282e-01, -2.28515139e-02,
	        -3.73130536e-01],
	       [ 2.51508536e-02,  1.37092782e-02, -3.03141246e-02,
	        -8.77395762e-02],
	       [ 2.54250392e-02, -1.80965332e-01, -3.20689161e-02,
	         1.95227190e-01],
	       [ 2.18057326e-02, -3.75614233e-01, -2.81643723e-02,
	         4.77623775e-01],
	       [ 1.42934479e-02, -1.80106189e-01, -1.86118968e-02,
	         1.76198783e-01],
	       [ 1.06913241e-02, -3.74956906e-01, -1.50879211e-02,
	         4.62952654e-01],
	       [ 3.19218599e-03, -1.79625013e-01, -5.82886806e-03,
	         1.65552483e-01],
	       [-4.00314264e-04,  1.55798899e-02, -2.51781839e-03,
	        -1.28963574e-01],
	       [-8.87164650e-05,  2.10737819e-01, -5.09708987e-03,
	        -4.22439777e-01],
	       [ 4.12603991e-03,  1.56884494e-02, -1.35458854e-02,
	        -1.31368062e-01],
	       [ 4.43980890e-03,  2.11001795e-01, -1.61732466e-02,
	        -4.28293569e-01],
	       [ 8.65984479e-03,  4.06349021e-01, -2.47391180e-02,
	        -7.26030813e-01],
	       [ 1.67868252e-02,  2.11577719e-01, -3.92597343e-02,
	        -4.41235819e-01],
	       [ 2.10183796e-02,  1.70327059e-02, -4.80844507e-02,
	        -1.61182560e-01],
	       [ 2.13590337e-02,  2.12808882e-01, -5.13081019e-02,
	        -4.68638779e-01],
	       [ 2.56152113e-02,  1.84478903e-02, -6.06808775e-02,
	        -1.92558916e-01],
	       [ 2.59841691e-02,  2.14383040e-01, -6.45320558e-02,
	        -5.03749939e-01],
	       [ 3.02718300e-02,  4.10352312e-01, -7.46070546e-02,
	        -8.16051249e-01],
	       [ 3.84788762e-02,  2.16326819e-01, -9.09280795e-02,
	        -5.47736755e-01],
	       [ 4.28054126e-02,  4.12600645e-01, -1.01882815e-01,
	        -8.67627284e-01],
	       [ 5.10574255e-02,  6.08950343e-01, -1.19235360e-01,
	        -1.19052631e+00],
	       [ 6.32364323e-02,  4.15557983e-01, -1.43045886e-01,
	        -9.37469623e-01],
	       [ 7.15475920e-02,  6.12288772e-01, -1.61795279e-01,
	        -1.27146402e+00],
	       [ 8.37933674e-02,  8.09063118e-01, -1.87224559e-01,
	        -1.61013118e+00]] * batch_size
		x = tf.convert_to_tensor(x)[-12:]
		x = (x - [(-2.3997673174407006, 2.39642385496039, -2.8818372014886324, 2.592270558555229, -0.20924535244879322, 0.2092951802278316, -2.103534190617424, 2.6894295591474777)])
				   -2.396124049322869 	2.398100575323115 -2.9579621705952297  2.7472035487282644 -0.20923500871729206 	0.20938613373324455 -1.7320768870117482 1.71236942339316
		# x = tf.expand_dims(x, 0)

		model.train_on_batch(x, Y)

	ipdb.set_trace()
	print(model.predict_on_batch(x))
	sys.exit()
test()


x = None
for episode in range(EPISODES):
	done = False
	state = env.reset()
	ep_rewards = 0
	states = []
	actions = []
	activations_history = []
	rewards = []
	dones = []
	values = []
	while not done:
		env.render()  
		# ipdb.set_trace()
		activations = tf.squeeze(model.predict_on_batch(tf.expand_dims(state, 0)))
		probs = tf.nn.softmax(activations)
		action = tf.squeeze(tf.random.categorical(tf.expand_dims(probs, 0), 1)).numpy()
		# action = np.argmax(probs)
		state_, reward, done, _ = env.step(action)
		if done:
			reward = 0

		states.append(state)
		actions.append(action)
		activations_history.append(activations)
		rewards.append(reward)
		dones.append(done)
		# values.append([left, right][action])

		ep_rewards += reward * (1-int(done))
		if not done:
			state = state_

	print(f"episode: {episode}, ep_rewards: {ep_rewards}")

	states = np.array(states)
	actions = np.array(actions)
	activations_history = np.array(activations_history)
	rewards = np.array(rewards)
	dones = np.array(dones)
	Y = getY(returns(rewards, dones).reshape(-1,1), actions, activations_history) #- np.squeeze(values)
	Y = tf.convert_to_tensor(Y)
	x = tf.convert_to_tensor(states)
	ipdb.set_trace()
	model.train_on_batch(x, Y)

ipdb.set_trace()

