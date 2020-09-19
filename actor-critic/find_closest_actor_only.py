import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import ipdb
import pygame

style.use("ggplot")
print(time.time())
np.random.seed(int(time.time()))

SIZE = 25
HM_EPISODES = SIZE*100
SHOW_EVERY = HM_EPISODES//HM_EPISODES  # how often to play through env visually.
REFRESH = 1

MOVE_PENALTY = -SIZE
FOOD_REWARD = 1
epsilon = 1.
EPS_DECAY = 0.998  # Every episode will be epsilon*EPS_DECAY

start_q_table = None # None or Filename

LEARNING_RATE = 0.01
DISCOUNT = 0.95

PLAYER_N = 1  # player key in dict
FOOD_N = 2  # food key in dict

# the dict!
d = {1: (255, 175, 0),
     2: (0, 255, 255),
     3: (0, 255, 255)}

gamma = 0.99


inputs = layers.Input(shape=(1,))
common = layers.Dense(128, activation="relu")(inputs)
output = layers.Dense(2)(common)
model = keras.Model(inputs=inputs, outputs=output)
model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'])

class Blob:
    def __init__(self, position=None):
        if position is not None:
            self.x = position
        else:
            self.x = np.random.randint(1, SIZE-1)
            print(self.x)

    def __str__(self):
        return f"{self.x}"

    def location(self):
        return self.x

    def action(self, choice):
        '''
        Gives us 2 total movement options. (0,1)
        '''
        if choice == 0:
            self.move(x=-1)
        elif choice == 1:
            self.move(x=1)

    def move(self, x):

        self.x += x

        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE-1:
            self.x = SIZE-1


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


pygame.init()
# steps = [0,1,0,0]
reward = MOVE_PENALTY
indicator_left = np.zeros((SIZE, 3), dtype=np.uint8)
indicator_right = np.zeros((SIZE, 3), dtype=np.uint8)
for episode in range(HM_EPISODES):
    player = Blob(position=np.random.randint(1, SIZE-1))
    food_1 = Blob(position=0)
    food_2 = Blob(position=SIZE-1)
    if episode % SHOW_EVERY == 0:
        show = True
    else:
        show = False

    episode_reward = 0
    done = False
    states = []
    actions = []
    activations_history = []
    rewards = []
    dones = []
    for i in range(SIZE*10):
        key_pressed = [i for i, k in enumerate(pygame.key.get_pressed()) if k==1]
        # print(key_pressed[79], key_pressed[80])
        if len(key_pressed) >0 and key_pressed[0] == 79:
            ipdb.set_trace()
        if show:
            env = np.zeros((4, SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
            env[0][food_1.x] = d[FOOD_N]  # sets the food location tile to green color
            env[0][food_2.x] = d[FOOD_N]
            env[0][player.x] = d[PLAYER_N]  # sets the player tile to blue

            env[2] = indicator_left
            env[3] = indicator_right

            img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
            img = img.resize((1600, 100), Image.NONE)  # resizing so we can see our agent in all its glory.
            cv2.imshow("game", np.array(img))  # show it!
            cv2.moveWindow("game", 0, 150)
            if reward == FOOD_REWARD:  # crummy code to hang at the end if we reach abrupt end for good reasons or not.
                if cv2.waitKey(REFRESH*1) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(REFRESH) & 0xFF == ord('q'):
                    break
        if done:
            break
        obs = player.location()/SIZE
        # GET THE ACTION
        # ipdb.set_trace()            
        left = np.array([1., 0.])
        right = np.array([0., 1.])
        activations = model.predict_on_batch(np.array([obs]))
        probs = tf.nn.softmax(tf.squeeze(np.array(activations)))
        # action = np.argmax(probs)
        action = tf.squeeze(tf.random.categorical(tf.expand_dims(probs, 0), 1)).numpy()
        # ipdb.set_trace()

        # if np.random.random() <= epsilon:
        #     action = np.random.randint(0, 2)
        

        # if player.location() == 2:
        #     action = 1
        # elif player.location() == SIZE-3:
        #     action = 0
        # elif player.location() == 1:
        #     # ipdb.set_trace()
        #     action = 0
        # elif player.location() == SIZE-2:
        #     # ipdb.set_trace()
        #     action = 1

        # Take the action!
        player.action(action)
        # player.action(steps[i%len(steps)])

        if player.x == food_1.x or player.x == food_2.x:
            reward = MOVE_PENALTY
            done = True
        else:
            reward = FOOD_REWARD
            done = False

        states.append(obs)
        actions.append(action)
        activations_history.append(activations[0])
        rewards.append(reward)
        dones.append(done)

    epsilon *= EPS_DECAY

    states = np.array(states)
    actions = np.array(actions)
    activations_history = np.array(activations_history)
    rewards = np.array(rewards)
    dones = np.array(dones)
    print(f"episode {episode}, rewards {np.sum(rewards)}, epsilon {epsilon}")
    # if action == 1 and reward > 0:
    #     ipdb.set_trace()
    Y = getY(returns(rewards, dones).reshape(-1,1), actions, activations_history) #- np.squeeze(values)
    Y = tf.convert_to_tensor(Y)
    x = states.reshape(-1,1)
    x = tf.convert_to_tensor(x)
    # ipdb.set_trace()
    model.train_on_batch(x, Y)
    # ipdb.set_trace()
    for i, cell in enumerate(env[2]):
        # ipdb.set_trace()
        activations = model.predict_on_batch(np.array([i/SIZE]))
        probs = tf.nn.softmax(tf.squeeze(np.array(activations)))
        # env[2][i] = (probs[0], 0, probs[1])
        indicator_left[i] = (0, probs[0]*255, 0)
        indicator_right[i] = (0, 0, probs[1]*255)
ipdb.set_trace()
# one hot action input
