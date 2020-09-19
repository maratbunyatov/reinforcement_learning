import gym
import numpy as np
import sys
import math
import time
from PIL import Image
import cv2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras

# np.set_printoptions(threshold=sys.maxsize)

np.set_printoptions(formatter={'float': '{: 0.2f}'.format}, suppress=True)

# env = gym.make("MountainCar-v0")
env = gym.make("Breakout-ram-v0")


ALPHA = 0.1
GAMMA = 1
EXPLORE = False

EPISODES = 10_000
SHOW_EVERY = 1
SLEEP = .0#0025
PRINT = False

PADDLE_IDX = 0#72
X_IDX = 1#99
Y_IDX = 2#101
MEM_ADDR = [72,99,101] #[70,99,] #[70,72,90,99,101,105]
# MEM_ADDR = [n for n in range(128)] 
# MEM_ADDR = [70,72,90,99,101,105]
ACTIONS_EXCLUDE = 1

INPUT_SIZE = len(MEM_ADDR)
HIDDEN_SIZE = 16
OUTPUT_SIZE = env.action_space.n-ACTIONS_EXCLUDE

# Exploration settings
epsilon = .0  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = 50
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)


# initialize model
model = Sequential()
# model.add(Dense(INPUT_SIZE, activation="sigmoid"))
model.add(Dense(HIDDEN_SIZE, activation="relu", input_dim=INPUT_SIZE))
# model.add(Dense(HIDDEN_SIZE, activation="relu"))
output_layer = Dense(OUTPUT_SIZE, activation="linear")
model.add(output_layer)
# compile the keras model
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(learning_rate=0.1), metrics=['accuracy'])

hidden_layers = keras.backend.function(
    [model.layers[0].input],  # we will feed the function with the input of the x layer  
    [model.layers[len(model.layers)-1].output,] # we want to get the output of the y layer
)

def s2x(s):
    result = []
    for i in range(len(s)):
        result.append(s[i] / 255)
    # result.append(feature*feature / (255**2))
    # result.append(abs(s[PADDLE_IDX]-s[X_IDX])**2 / 255**2)
    # result.append(0 if (s[PADDLE_IDX]-s[X_IDX]) / 255 <= 0 else 1)    
    return np.array(result)

episode_reward = 0
prev_ep_rewards = -200
rewards = []
reward_eligible = False
penalty_eligible = False
step_counter = 0
for episode in range(EPISODES):
    print(episode, step_counter)
    s_t1 = np.array(env.reset())[MEM_ADDR]
    done = False

    if episode and not episode % SHOW_EVERY:
        render = True
        print(episode, step_counter, np.mean(rewards[-SHOW_EVERY:]), f"{np.std(rewards[-SHOW_EVERY:]):.2f}", int(np.min(rewards[-SHOW_EVERY:])), int(np.max(rewards[-SHOW_EVERY:])), 
            # f" ...  (theta: \n{model.theta[:,[PADDLE_IDX,X_IDX,Y_IDX]]})"
            # f" ...  (w_1_2: \n{model.w_1_2})\n (w_2_3: \n{model.w_2_3})\nQ: [{Q_t1[2]:.2f}, {Q_t1[0]:.2f}, {Q_t1[1]:.2f}] Weights: [{theta_sum[2]:.2f}, {theta_sum[0]:.2f}, {theta_sum[1]:.2f}]"
        )
    else:
        render = False

    rewards.append(episode_reward)
    episode_reward = 0
    left_right = 0
    up_down = 0
    while not done:
        step_counter += 1

        # Peceive
        Q_t1 = model.predict(s2x(s_t1).reshape(1,INPUT_SIZE))[0]

        # Act, Reward
        if np.random.random() > epsilon:  
            a = np.argmax(Q_t1)
        else:
            a = np.random.randint(0, OUTPUT_SIZE)
        if s_t1[Y_IDX] == 0:
            a = 1 - ACTIONS_EXCLUDE
        s_t2, reward, done, _ = env.step(a+ACTIONS_EXCLUDE)
        s_t2 = np.array(s_t2)[MEM_ADDR]
        episode_reward += reward
        if PRINT: print(a+ACTIONS_EXCLUDE, s_t2, reward)

        r = 0
        if reward_eligible and s_t1[Y_IDX] > s_t2[Y_IDX] and s_t2[Y_IDX] > 150:
            reward_eligible = False
            r = 1
        elif not reward_eligible and s_t2[Y_IDX] < 150:
            reward_eligible = True
        elif s_t2[Y_IDX] > 185 and penalty_eligible:
            penalty_eligible = False
            r = -100
        elif not penalty_eligible and s_t2[Y_IDX] < 185:
            penalty_eligible = True

        # Peceive'               
        Q_t2 = model.predict(s2x(s_t2).reshape(1,INPUT_SIZE))[0]
        max_value_t2 = np.max(Q_t2)

        # Learn
        if 1==2:#done:
            if episode_reward > prev_ep_rewards:
                prev_ep_rewards = episode_reward
                print("improvement ", episode_reward, episode)

            Q_t1[a] = r
            model.fit(x=s_t1.reshape(1,INPUT_SIZE), y=Q_t1.reshape(1,OUTPUT_SIZE), verbose=0)
        else:
            # Back-propagate
            if r==1 or r==-100: print(f"Q: [{Q_t1[2]:.2f}, {Q_t1[0]:.2f}, {Q_t1[1]:.2f}]")
            Q_t1[a] = r + GAMMA*max_value_t2
            if r==1 or r==-100: print(f"Q: [{Q_t1[2]:.2f}, {Q_t1[0]:.2f}, {Q_t1[1]:.2f}]")
            model.fit(x=s2x(s_t1).reshape(1,INPUT_SIZE), y=Q_t1.reshape(1,OUTPUT_SIZE), verbose=0)

        s_t1 = s_t2

        if not episode % SHOW_EVERY:
            time.sleep(SLEEP)
            env.render()

            display = np.zeros((3, len(Q_t1), 3), dtype=np.uint8)            
            min_q = np.min(Q_t1)
            max_q = np.max(Q_t1)

            output = hidden_layers(s_t1.reshape(1,INPUT_SIZE))[0][0]
            min_q = np.min(output)
            max_q = np.max(output)
            left = ((output[2] - min_q)*255)/(max_q - min_q)  if max_q != min_q else 0
            nothing = ((output[0] - min_q)*255)/(max_q - min_q)  if max_q != min_q else 0
            right = ((output[1] - min_q)*255)/(max_q - min_q) if max_q != min_q else 0
            display[0][0] = (left, 0, 0)
            display[0][1] = (0, nothing, 0)
            display[0][2] = (0, 0, right)

            theta_sum = output_layer.get_weights()[0].sum(axis=0) + output_layer.get_weights()[1]
            # print(f"Weights: [{theta_sum[2]:.2f}, {theta_sum[0]:.2f}, {theta_sum[1]:.2f}]",
            #     f"hidden_layers[{hidden_layers(s_t1.reshape(1,INPUT_SIZE))}]")
            display[2][0] = ((theta_sum[2] - theta_sum.min())*255 / (theta_sum.max() - theta_sum.min()), 0, 0)
            display[2][1] = (0, (theta_sum[0] - theta_sum.min())*255 / (theta_sum.max() - theta_sum.min()), 0)
            display[2][2] = (0, 0, (theta_sum[1] - theta_sum.min())*255 / (theta_sum.max() - theta_sum.min()))

            img = Image.fromarray(display, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
            img = img.resize((300, 50), Image.NONE)  # resizing so we can see our agent in all its glory.
            cv2.imshow("Q", np.array(img))  # show it!

            # cv2.moveWindow("Q", 0, 150)
            # if reward == FOOD_REWARD:  # crummy code to hang at the end if we reach abrupt end for good reasons or not.
            #     if cv2.waitKey(500) & 0xFF == ord('q'):
            #         break
            # else:
            #     if cv2.waitKey(REFRESH) & 0xFF == ord('q'):
            #         break

        # sys.exit()
    # Decaying is being done every episode if episode number is within decaying range
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value


env.close()