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

SIZE = 100
HM_EPISODES = SIZE*80
SHOW_EVERY = HM_EPISODES//10  # how often to play through env visually.
REFRESH = 10

ALPHA = 0.01#1/(HM_EPISODES/100)
GAMMA = 1

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


class Model:
  def __init__(self):
    self.theta = np.random.randn(2,3)

  def s2x(self, s):
    return np.array([
      s / SIZE          ,
      s*s / (SIZE**2)   ,
      1
    ])

  def perceive(self, s):
    x = self.s2x(s)
    return self.theta.dot(x)

  def gradient(self, s):
    return self.s2x(s)


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

# initialize model
model = Model()

for episode in range(HM_EPISODES):
    player = Blob(position=np.random.randint(1, SIZE-1))
    food_1 = Blob(position=0)
    food_2 = Blob(position=SIZE-1)

    if episode % SHOW_EVERY == 0:
        print(f"#{episode}, mean: {np.mean(episode_rewards[-SHOW_EVERY:])}, epsilon: {epsilon}, theta: {model.theta}")
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(SIZE//2):
        
        # Peceive
        s_t1 = player.location()
        Q_t1 = model.perceive(s_t1)

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
        Q_t2 = model.perceive(s_t2)
        max_q_t2 = np.max(Q_t2)
        value_t1 = Q_t1[a]

        # Learn
        if reward == FOOD_REWARD:
            model.theta[a] += ALPHA*(reward - value_t1)*model.gradient(s_t1)
        else:
            model.theta[a] += ALPHA*(reward + GAMMA*max_q_t2 - value_t1)*model.gradient(s_t1)


        # Display
        if show or episode==HM_EPISODES-1:
            env = np.zeros((4, SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
            env[0][food_1.x] = d[FOOD_N]  # sets the food location tile to green color
            env[0][food_2.x] = d[FOOD_N]
            env[0][player.x] = d[PLAYER_N]  # sets the player tile to blue
            
            q_s = []
            for i in range(SIZE):
                q = model.perceive(i)
                q_s.extend(q)
            min_q = min(q_s)
            max_q = max(q_s)
            for i in range(SIZE):
                right = ((model.perceive(i)[0] - min_q)*255)/(max_q - min_q) if max_q != min_q else 0
                left = ((model.perceive(i)[1] - min_q)*255)/(max_q - min_q)  if max_q != min_q else 0
                env[2][i] = (0, right, left)
                
                if left > right:    
                    env[3][i] = (0, 0, 255)
                else:  
                    env[3][i] = (0, 255, 0)

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