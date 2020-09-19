import numpy as np
import sys
import time
from PIL import Image
import cv2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras

np.set_printoptions(formatter={'float': '{: 0.2f}'.format}, suppress=True)

EPISODES = 2_000
SHOW_EVERY = 100

INPUT_SIZE = 2
HIDDEN_SIZE = 16
OUTPUT_SIZE = 2


# initialize model
model = Sequential()
# model.add(Dense(INPUT_SIZE, activation="sigmoid"))
model.add(Dense(HIDDEN_SIZE, activation="relu", input_dim=INPUT_SIZE))
# model.add(Dense(HIDDEN_SIZE, activation="relu"))
# model.add(Dense(HIDDEN_SIZE, activation="relu"))
# model.add(Dense(HIDDEN_SIZE, activation="relu"))
model.add(Dense(OUTPUT_SIZE, activation="sigmoid"))
# compile the keras model
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])

episode_reward = 0
prev_ep_rewards = -200
errors = []
states_counter = 0

inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])

answers = {}
answers["[0 0]"] = np.array([1, 0])
answers["[1 1]"] = np.array([1, 0])
answers["[1 0]"] = np.array([0, 1])
answers["[0 1]"] = np.array([0, 1])

for episode in range(EPISODES):
    s_t1 = inputs[np.random.randint(len(inputs))]

    # Back-propagate  
    model.fit(x=s_t1.reshape(1,INPUT_SIZE), y=answers[str(s_t1)].reshape(1,OUTPUT_SIZE), verbose=0)

    if episode and not episode % SHOW_EVERY:
        render = True
        print(episode, np.mean(errors[-SHOW_EVERY:]), f"{np.std(errors[-SHOW_EVERY:]):.2f}", int(np.sum(errors[-SHOW_EVERY:])), 
            # f" ...  (theta: \n{model.theta[:,[PADDLE_IDX,X_IDX,Y_IDX]]})"
            # f" ...  (w_1_2: \n{model.w_1_2})\n (w_2_3: \n{model.w_2_3})", 
            # f"\nQ: {Q_t1:.2f} Weights: [{theta_sum[2]:.2f}, {theta_sum[0]:.2f}, {theta_sum[1]:.2f}]"
        )
        errors = []

    # Peceive
    Q_t1 = model.predict(s_t1.reshape(1,INPUT_SIZE))
    # print(f"Q: [{Q_t1[2]:.2f}, {Q_t1[0]:.2f}, {Q_t1[1]:.2f}] Weights: [{theta_sum[2]:.2f}, {theta_sum[0]:.2f}, {theta_sum[1]:.2f}]")

    # Learn
    e = answers[str(s_t1)] - Q_t1
    # print(f"episode: {episode}, Q_t1: {Q_t1}, target: {answers[str(s_t1)]}, error: {e}")
    errors.append(abs(e))
    if episode > EPISODES-10:
        print(f"s_t1:{s_t1}, Q_t1:{Q_t1}, target:{answers[str(s_t1)]}, error:{e}")


    # evaluate the keras model
    # _, accuracy = model.ann.evaluate(X, y)
    # print('Accuracy: %.2f' % (accuracy*100))


    