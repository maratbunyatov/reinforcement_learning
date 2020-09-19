import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras

style.use("ggplot")
print(time.time())
np.random.seed(int(time.time()))

SIZE = 100
HM_EPISODES = SIZE*8
SHOW_EVERY = HM_EPISODES//10  # how often to play through env visually.
REFRESH = 0

ALPHA = 0.01#1/(HM_EPISODES/100)
GAMMA = 1

INPUT_SIZE = 1
HIDDEN_SIZE = 32
OUTPUT_SIZE = 2

MOVE_PENALTY = -1
FOOD_REWARD = SIZE//SIZE
epsilon = 0.00009
EPS_DECAY = 0.9998  # Every episode will be epsilon*EPS_DECAY

PLAYER_N = 1  # player key in dict
FOOD_N = 2  # food key in dict
ACTION_SPACE = ('R', 'L')


d = {1: (255, 175, 0),
     2: (0, 255, 255),
     3: (0, 255, 255)}

# initialize model
model = Sequential()
# model.add(Dense(INPUT_SIZE, activation="sigmoid"))
# model.add(Dense(HIDDEN_SIZE, activation="relu", input_dim=INPUT_SIZE))
# model.add(Dense(HIDDEN_SIZE, activation="relu"))
output_layer = Dense(OUTPUT_SIZE, activation="linear", input_dim=INPUT_SIZE)
model.add(output_layer)
# compile the keras model
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(learning_rate=.1), metrics=['accuracy'])

# hidden_layers = keras.backend.function(
#     [model.layers[0].input],  # we will feed the function with the input of the x layer  
#     [model.layers[len(model.layers)-1].output,] # we want to get the output of the y layer
# )


def s2x(s):
    # result = []
    # for i in range(len(s)):
    #     result.append(s[i] / SIZE)
    # result.append(feature*feature / (255**2))
    # result.append(abs(s[PADDLE_IDX]-s[X_IDX])**2 / 255**2)
    # result.append(0 if (s[PADDLE_IDX]-s[X_IDX]) / 255 <= 0 else 1)    
    return np.array([s])


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
            self.move(x=1)
        elif choice == 1:
            self.move(x=-1)

    def move(self, x):

        self.x += x

        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE-1:
            self.x = SIZE-1



episode_rewards = []

for episode in range(HM_EPISODES):
    player = Blob(position=np.random.randint(1, SIZE-1))
    food_1 = Blob(position=0)
    food_2 = Blob(position=SIZE-1)

    if episode % SHOW_EVERY == 0:
        print(f"#{episode}, mean: {np.mean(episode_rewards[-SHOW_EVERY:])}, epsilon: {epsilon}")
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(SIZE//2):

        # Peceive
        s_t1 = player.location()
        Q_t1 = model.predict(s2x(s_t1))[0]

        # Act
        if np.random.random() > epsilon:  
            a = np.argmax(Q_t1)
        else:
            a = np.random.randint(0, len(ACTION_SPACE))
        player.action(a)

        # Reward        
        s_t2 = player.location()
        if player.x == food_1.x or player.x == food_2.x:
            reward = FOOD_REWARD
        else:
            reward = MOVE_PENALTY
        
        # Peceive'               
        Q_t2 = model.predict(s2x(s_t2))[0]
        max_q_t2 = np.max(Q_t2) 
        value_t1 = Q_t1[a]
        Q_t1_train = np.copy(Q_t1)       
        Q_t1_train[a] = reward + GAMMA*max_q_t2

        # Learn
        if reward == FOOD_REWARD:
            model.fit(x=s2x(s_t1).reshape(1,INPUT_SIZE), y=Q_t1_train.reshape(1,OUTPUT_SIZE), verbose=0)
            # model.theta[a] += ALPHA*(reward - value_t1)*model.gradient(s_t1)
        else:
            model.fit(x=s2x(s_t1).reshape(1,INPUT_SIZE), y=Q_t1_train.reshape(1,OUTPUT_SIZE), verbose=0)
            # model.theta[a] += ALPHA*(reward + GAMMA*max_q_t2 - value_t1)*model.gradient(s_t1)


        # Display
        if show or episode==HM_EPISODES-1:
            env = np.zeros((4, SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
            env[0][food_1.x] = d[FOOD_N]  # sets the food location tile to green color
            env[0][food_2.x] = d[FOOD_N]
            env[0][player.x] = d[PLAYER_N]  # sets the player tile to blue
            
            # q_s = []
            # for i in range(SIZE):
            #     q = model.predict(s2x(i))[0]
            #     q_s.extend(q)
            # min_q = min(q_s)
            # max_q = max(q_s)
            # for i in range(SIZE):
            #     pred = model.predict(s2x(i))[0]
            #     right = ((pred[0] - min_q)*255)/(max_q - min_q) if max_q != min_q else 0
            #     left = ((pred[1] - min_q)*255)/(max_q - min_q)  if max_q != min_q else 0
            #     env[2][i] = (0, right, left)
                
            #     if left > right:    
            #         env[3][i] = (0, 0, 255)
            #     else:  
            #         env[3][i] = (0, 255, 0)

            img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
            img = img.resize((1600, 100), Image.NONE)  # resizing so we can see our agent in all its glory.
            cv2.imshow("game", np.array(img))  # show it!
            cv2.moveWindow("game", 0, 150)
            cv2.waitKey(1)
            # if reward == FOOD_REWARD:  # crummy code to hang at the end if we reach abrupt end for good reasons or not.
            #     if cv2.waitKey(500) & 0xFF == ord('q'):
            #         break
            # else:
            #     if cv2.waitKey(REFRESH) & 0xFF == ord('q'):
            #         break

        episode_reward += reward
        if reward == FOOD_REWARD:
            break

    #print(episode_reward)
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

# cv2.waitKey(50000)

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

# with open(f"qtables/qtable-{int(time.time())}.pickle", "wb") as f:
#     pickle.dump(q_table, f)