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
HM_EPISODES = SIZE*1000
SHOW_EVERY = HM_EPISODES//100  # how often to play through env visually.
REFRESH = 5

MOVE_PENALTY = -1
FOOD_REWARD = SIZE//2
epsilon = 0.00009
EPS_DECAY = 0.9998  # Every episode will be epsilon*EPS_DECAY

start_q_table = None # None or Filename

ALPHA = 1/(HM_EPISODES/100)
GAMMA = 1

PLAYER_N = 1  # player key in dict
FOOD_N = 2  # food key in dict
ACTION_SPACE = ('R', 'L')

# x*10.24383144  x*x20.87576475  -3.32446794 x*-16.43005386  -3.76555477   16.39346978  12.29327354
class Model:
  def __init__(self):
    self.theta = np.random.randn(7) / np.sqrt(7)
    # if we use SA2IDX, a one-hot encoding for every (s,a) pair
    # in reality we wouldn't want to do this b/c we have just
    # as many params as before
    # print "D:", IDX
    # self.theta = np.random.randn(IDX) / np.sqrt(IDX)

  def sa2x(self, s, a):
    # NOTE: using just (r, c, r*c, u, d, l, r, 1) is not expressive enough
    return np.array([
      s / SIZE          if a == 'R' else 0,
      s*s / (SIZE**2)   if a == 'R' else 0,
      1                 if a == 'R' else 0,
      s / SIZE          if a == 'L' else 0,
      s*s / (SIZE**2)   if a == 'L' else 0,
      1                 if a == 'L' else 0,
      1
    ])
    # if we use SA2IDX, a one-hot encoding for every (s,a) pair
    # in reality we wouldn't want to do this b/c we have just
    # as many params as before
    # x = np.zeros(len(self.theta))
    # idx = SA2IDX[s][a]
    # x[idx] = 1
    # return x

  def predict(self, s, a):
    x = self.sa2x(s, a)
    return self.theta.dot(x)

  def grad(self, s, a):
    return self.sa2x(s, a)


def getQs(model, s):
  # we need Q(s,a) to choose an action
  # i.e. a = argmax[a]{ Q(s,a) }
  Qs = {}
  for a in ACTION_SPACE:
    q_sa = model.predict(s, a)
    Qs[a] = q_sa
  return Qs


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
        if choice == 'R':
            self.move(x=1)
        elif choice == 'L':
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

def max_dict(d):
    # returns the argmax (key) and max (value) from a dictionary
    # put this into a function since we are using it so often
    max_key = None
    max_val = float('-inf')
    for k, v in d.items():
        if v > max_val:
            max_val = v
            max_key = k
    return max_key, max_val

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
        
        s1 = player.location()

        # get Q(s) so we can choose the first action
        Q1 = getQs(model, s1)

        #print(obs)
        if np.random.random() > epsilon:            
            # the first (s, r) tuple is the state we start in and 0
            # (since we don't get a reward) for simply starting the game
            # the last (s, r) tuple is the terminal state and the final reward
            # the value for the terminal state is by definition 0, so we don't
            # care about updating it.
            a1 = max_dict(Q1)[0]
        else:
            a1 = np.random.randint(0, 2)
        # Take the action!
        player.action(a1)

        s2 = player.location()

        #### MAYBE ###
        # enemy.move()
        # food.move()
        ##############

        if player.x == food_1.x or player.x == food_2.x:
            reward = FOOD_REWARD
        else:
            reward = MOVE_PENALTY
        ## NOW WE KNOW THE REWARD, LET'S CALC YO
        # first we need to obs immediately after the move.
        max_future_q = max_dict(getQs(model, s2))[1]
        # current_q = Qs[a]

        if reward == FOOD_REWARD:
            model.theta += ALPHA*(reward - model.predict(s1, a1))*model.grad(s1, a1)
        else:
            # we will update Q(s,a) AS we experience the episode
            model.theta += ALPHA*(reward + GAMMA*max_future_q - model.predict(s1, a1))*model.grad(s1, a1)
        # new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        if show:
            env = np.zeros((4, SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
            env[0][food_1.x] = d[FOOD_N]  # sets the food location tile to green color
            env[0][food_2.x] = d[FOOD_N]
            env[0][player.x] = d[PLAYER_N]  # sets the player tile to blue
            
            q_s = []
            for i in range(SIZE):
                q = getQs(model, i)
                q_s.append(q['R'])
                q_s.append(q['L'])
            min_q = min(q_s)
            max_q = max(q_s)
            for i in range(SIZE):
                right = ((getQs(model, i)['R'] - min_q)*255)/(max_q - min_q) if max_q != min_q else 0
                left = ((getQs(model, i)['L'] - min_q)*255)/(max_q - min_q)  if max_q != min_q else 0
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
                if cv2.waitKey(100) & 0xFF == ord('q'):
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

with open(f"qtables/qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)