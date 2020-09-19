import gym
import numpy as np
import sys
import math
import time
from PIL import Image
import cv2

# np.set_printoptions(threshold=sys.maxsize)

np.set_printoptions(formatter={'float': '{: 0.2f}'.format}, suppress=True)

# env = gym.make("MountainCar-v0")
env = gym.make("Breakout-ram-v0")


ALPHA = 0.1
GAMMA = 1
EXPLORE = False

EPISODES = 10000
SHOW_EVERY = 1
SLEEP = .0025
PRINT = False

PADDLE_IDX = 0#72
X_IDX = 1#99
Y_IDX = 2#101
MEM_ADDR = [72,99,101] #[70,99,] #[70,72,90,99,101,105]
# MEM_ADDR = [n for n in range(128)] 
# MEM_ADDR = [70,72,90,99,101,105]
ACTIONS_EXCLUDE = 1

# Exploration settings
epsilon = .0  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = 50
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)


class Model:
  def __init__(self):
    self.theta = np.random.randn(env.action_space.n-ACTIONS_EXCLUDE, len(MEM_ADDR)*1+0) * np.sqrt(1/(len(MEM_ADDR)*1+0)) # Xavier intitialization 

  def s2x(self, s):
    # return s / 255
    result = []
    # for i in range(len(s)):
    #   result.append(feature / 255)
      # result.append(feature*feature / (255**2))
    result.append(s[2] / 255)
    result.append(1)
    # result.append(abs(s[PADDLE_IDX]-s[X_IDX])**2 / 255**2)
    result.append(0 if (s[PADDLE_IDX]-s[X_IDX]) / 255 <= 0 else 1)
    
    return np.array(result)

  def perceive(self, s):
    x = self.s2x(s)
    return self.activation(self.theta.dot(x))

  def gradient(self, s):
    return self.s2x(s)

  def activation(self, z):
    # return z
    # Softmax
    # z -= np.max(z)
    # sm = (np.exp(z).T / np.sum(np.exp(z), axis=0)).T
    sm = (np.exp(z) / np.sum(np.exp(z)))
    return sm

    # Sigmoid
    # return 1 / (1 + np.exp(-x))

  def activation_gradient(self, s):
    # return np.eye(s.shape[0], s.shape[0])
    # Softmax
    # Take the derivative of softmax element w.r.t the each logit which is usually Wi * X
    # input s is softmax value of the original input x. 
    # s.shape = (1, n) 
    # i.e. s = np.array([0.3, 0.7]), x = np.array([0, 1])
    # initialize the 2-D jacobian matrix.
    jacobian_m = np.diag(s)
    for i in range(len(jacobian_m)):
        for j in range(len(jacobian_m)):
            if i == j:
                jacobian_m[i][j] = s[i] * (1-s[i])
            else: 
                jacobian_m[i][j] = -s[i]*s[j]
    return jacobian_m



    # Sigmoid
    # return self.activation(x) * (1 - self.activation(x))


# initialize model
model = Model()

episode_reward = 0
prev_ep_rewards = -200
rewards = []
q_table = {}
states_counter = 0
eligible = False
for episode in range(EPISODES):
    s_t1 = env.reset().astype(int)[MEM_ADDR]
    done = False

    if episode and not episode % SHOW_EVERY:
        render = True
        print(episode, np.mean(rewards[-SHOW_EVERY:]), f"{np.std(rewards[-SHOW_EVERY:]):.2f}", int(np.min(rewards[-SHOW_EVERY:])), int(np.max(rewards[-SHOW_EVERY:])), 
            # f" ...  (theta: \n{model.theta[:,[PADDLE_IDX,X_IDX,Y_IDX]]})"
            f" ...  (theta: \n{model.theta}) \nQ: [{Q_t1[2]:.2f}, {Q_t1[0]:.2f}, {Q_t1[1]:.2f}] Weights: [{theta_sum[2]:.2f}, {theta_sum[0]:.2f}, {theta_sum[1]:.2f}]"
        )
    else:
        render = False

    rewards.append(episode_reward)
    episode_reward = 0
    step_counter = 0
    left_right = 0
    up_down = 0
    while not done:
        step_counter += 1

        # Peceive
        Q_t1 = model.perceive(s_t1)
        theta_sum = model.theta.sum(axis=1)
        # print(f"Q: [{Q_t1[2]:.2f}, {Q_t1[0]:.2f}, {Q_t1[1]:.2f}] Weights: [{theta_sum[2]:.2f}, {theta_sum[0]:.2f}, {theta_sum[1]:.2f}]")

        # Act, Reward
        if np.random.random() > epsilon:  
            a = np.argmax(Q_t1)
        else:
            a = np.random.randint(0, env.action_space.n-ACTIONS_EXCLUDE)
        if s_t1[Y_IDX] == 0:
            a = 1 - ACTIONS_EXCLUDE
        s_t2, reward, done, _ = env.step(a+ACTIONS_EXCLUDE)
        s_t2 = s_t2.astype(int)[MEM_ADDR] 
        episode_reward += reward
        if PRINT: print(a+ACTIONS_EXCLUDE, s_t2, reward)

        # Peceive'               
        Q_t2 = model.perceive(s_t2)
        max_value_t2 = np.max(Q_t2)
        value_t1 = Q_t1[a]

        # Learn
        r = 0
        if eligible and s_t1[Y_IDX] > s_t2[Y_IDX] and s_t2[Y_IDX] > 150:
            eligible = False
            r = 1
        elif not eligible and s_t2[Y_IDX] < 150:
            eligible = True
        elif s_t2[Y_IDX] == 0:
            r = -5

        if s_t1[Y_IDX] > 0:
            if done:
                if episode_reward > prev_ep_rewards:
                    prev_ep_rewards = episode_reward
                    print("improvement ", episode_reward, episode)

                # model.theta[a] += ALPHA*(r - value_t1) * model.activation_gradient(Q_t1)[a,a] * model.gradient(s_t1)
            else:
                model.theta[a] += ALPHA*(r + GAMMA*max_value_t2 - value_t1) * model.activation_gradient(Q_t1)[a,a] * model.gradient(s_t1)

        s_t1 = s_t2

        if not episode % SHOW_EVERY:
            time.sleep(SLEEP)
            env.render()

            display = np.zeros((3, len(Q_t1), 3), dtype=np.uint8)            
            min_q = np.min(Q_t1)
            max_q = np.max(Q_t1)

            left = ((Q_t1[2] - min_q)*255)/(max_q - min_q)  if max_q != min_q else 0
            nothing = ((Q_t1[0] - min_q)*255)/(max_q - min_q)  if max_q != min_q else 0
            right = ((Q_t1[1] - min_q)*255)/(max_q - min_q) if max_q != min_q else 0
            display[0][0] = (left, 0, 0)
            display[0][1] = (0, nothing, 0)
            display[0][2] = (0, 0, right)

            display[2][0] = ((model.theta[2].sum() - theta_sum.min())*255 / (theta_sum.max() - theta_sum.min()), 0, 0)
            display[2][1] = (0, (model.theta[0].sum() - theta_sum.min())*255 / (theta_sum.max() - theta_sum.min()), 0)
            display[2][2] = (0, 0, (model.theta[1].sum() - theta_sum.min())*255 / (theta_sum.max() - theta_sum.min()))

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