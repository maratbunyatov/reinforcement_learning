import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use("ggplot")
print(time.time())
np.random.seed(int(time.time()))

SIZE = 50
HM_EPISODES = SIZE*10
SHOW_EVERY = HM_EPISODES//10  # how often to play through env visually.
REFRESH = 100

MOVE_PENALTY = -1
FOOD_REWARD = SIZE//2
epsilon = 0.0009
EPS_DECAY = 0.9998  # Every episode will be epsilon*EPS_DECAY

start_q_table = None # None or Filename

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1  # player key in dict
FOOD_N = 2  # food key in dict

# the dict!
d = {1: (255, 175, 0),
     2: (0, 255, 255),
     3: (0, 255, 255)}


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


if start_q_table is None:
    # initialize the q-table#
    q_table = {}

else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

q_values_count = 0
def qvalue(obs):
    global q_values_count
    if obs in q_table: 
        return q_table[obs]  
    q_values_count += 1
    q_table[obs] = np.random.uniform(low=0, high=1, size=2)
    return q_table[obs]

episode_rewards = []

for episode in range(HM_EPISODES):
    player = Blob(position=np.random.randint(1, SIZE-1))
    food_1 = Blob(position=0)
    food_2 = Blob(position=SIZE-1)
    if episode % SHOW_EVERY == 0:
        print(f"#{episode}, mean: {np.mean(episode_rewards[-SHOW_EVERY:])}, epsilon: {epsilon}, qvalues: {q_values_count}")
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(SIZE//2):
        obs = player.location()
        #print(obs)
        if np.random.random() > epsilon:
            # GET THE ACTION
            action = np.argmax(qvalue(obs))
        else:
            action = np.random.randint(0, 2)
        # Take the action!
        player.action(action)

        #### MAYBE ###
        #enemy.move()
        #food.move()
        ##############

        if player.x == food_1.x or player.x == food_2.x:
            reward = FOOD_REWARD
        else:
            reward = MOVE_PENALTY
        ## NOW WE KNOW THE REWARD, LET'S CALC YO
        # first we need to obs immediately after the move.
        new_obs = player.location()
        max_future_q = np.max(qvalue(new_obs))
        current_q = qvalue(new_obs)[action]

        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        qvalue(obs)
        q_table[obs][action] = new_q

        if show:
            env = np.zeros((3, SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
            env[0][food_1.x] = d[FOOD_N]  # sets the food location tile to green color
            env[0][food_2.x] = d[FOOD_N]
            env[0][player.x] = d[PLAYER_N]  # sets the player tile to blue
            q_s = []
            for k, v in q_table.items():
                q_s.append(v[0])
                q_s.append(v[1])
            # print(q_s)
            min_q = min(q_s)
            max_q = max(q_s)
            for k, v in q_table.items():
                if len(q_s) <= 1:
                    break
                right = ((v[0] - min_q)*255)/(max_q - min_q)
                left = ((v[1] - min_q)*255)/(max_q - min_q)
                # if right > left:
                #     left /= 8
                # else:
                #     right /= 8
                # print(right, left, q_max, q_min, v[0], v[1], k)
                env[2][k] = (left, 0, right) 

            img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
            img = img.resize((1600, 100), Image.NONE)  # resizing so we can see our agent in all its glory.
            cv2.imshow("game", np.array(img))  # show it!
            cv2.moveWindow("game", 0, 150)
            if reward == FOOD_REWARD:  # crummy code to hang at the end if we reach abrupt end for good reasons or not.
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(REFRESH) & 0xFF == ord('q'):
                    break

        episode_reward += reward
        if reward == FOOD_REWARD:
            q_table[new_obs][action] = new_q
            break

    #print(episode_reward)
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

cv2.waitKey(50000)
# moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')
# plt.plot([i for i in range(len(moving_avg))], moving_avg)
# plt.ylabel(f"Reward {SHOW_EVERY}ma")
# plt.xlabel("episode #")
# plt.show()

with open(f"qtables/qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)