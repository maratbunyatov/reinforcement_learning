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
HIDDEN_SIZE = 4
OUTPUT_SIZE = env.action_space.n-ACTIONS_EXCLUDE

# Exploration settings
epsilon = .0  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = 50
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

class Model:
  def __init__(self):
    self.l_2 = np.append(np.random.randn(HIDDEN_SIZE) / np.sqrt(HIDDEN_SIZE), 1)

    self.w_1_2 = np.random.randn(INPUT_SIZE, HIDDEN_SIZE) / np.sqrt(INPUT_SIZE) # Xavier intitialization 
    self.w_1_2_b = np.random.randn(HIDDEN_SIZE) / np.sqrt(INPUT_SIZE) # bias weights
    self.w_2_3 = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) / np.sqrt(HIDDEN_SIZE) 
    self.w_2_3_b = np.random.randn(OUTPUT_SIZE) / np.sqrt(HIDDEN_SIZE) 

  def s2x(self, s):
    result = []
    for i in range(len(s)):
      result.append(s[i] / 255)
      # result.append(feature*feature / (255**2))
    # result.append(abs(s[PADDLE_IDX]-s[X_IDX])**2 / 255**2)
    # result.append(0 if (s[PADDLE_IDX]-s[X_IDX]) / 255 <= 0 else 1)    
    return np.array(result)

  def perceive(self, s):
    x = self.s2x(s)    
    # forward propagate
    self.l_2 = self.sigmoid_activation(self.w_1_2.T.dot(x) + self.w_1_2_b)
    l3_activations = self.sigmoid_activation(self.w_2_3.T.dot(self.l_2) + self.w_2_3_b)
    return l3_activations

  def input_gradient(self, s):
    return self.s2x(s)

  def softmax_activation(self, z):
    # Softmax
    # z -= np.max(z)
    # sm = (np.exp(z).T / np.sum(np.exp(z), axis=0)).T
    sm = (np.exp(z) / np.sum(np.exp(z)))
    return sm

  def softmax_gradient(self, s):
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

  def sigmoid_activation(self, input):
    # Sigmoid
    return 1 / (1 + np.exp(-input))

  def sigmoid_gradient(self, s):
    # sigmoid(x) * (1 - sigmoid(x));, Assumes s is the sigmoid value
    return s * (1 - s)


# initialize model
model = Model()

episode_reward = 0
prev_ep_rewards = -200
rewards = []
eligible = False
step_counter = 0
for episode in range(EPISODES):
    s_t1 = env.reset().astype(int)[MEM_ADDR]
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
        Q_t1 = model.perceive(s_t1)
        theta_sum = model.w_2_3.sum(axis=0)
        # print(f"Q: [{Q_t1[2]:.2f}, {Q_t1[0]:.2f}, {Q_t1[1]:.2f}] Weights: [{theta_sum[2]:.2f}, {theta_sum[0]:.2f}, {theta_sum[1]:.2f}]")

        # Act, Reward
        if np.random.random() > epsilon:  
            a = np.argmax(Q_t1)
        else:
            a = np.random.randint(0, OUTPUT_SIZE)
        if s_t1[Y_IDX] == 0:
            a = 1 - ACTIONS_EXCLUDE
        s_t2, reward, done, _ = env.step(a+ACTIONS_EXCLUDE)
        s_t2 = s_t2.astype(int)[MEM_ADDR] 
        episode_reward += reward
        if PRINT: print(a+ACTIONS_EXCLUDE, s_t2, reward)

        r = 0
        if eligible and s_t1[Y_IDX] > s_t2[Y_IDX] and s_t2[Y_IDX] > 150:
            eligible = False
            r = 1
        elif not eligible and s_t2[Y_IDX] < 150:
            eligible = True
        elif s_t2[Y_IDX] == 0:
            r = -10

        # Peceive'               
        Q_t2 = model.perceive(s_t2)
        max_value_t2 = np.max(Q_t2)
        value_t1 = Q_t1[a]

        # Learn
        if s_t1[Y_IDX] > 0:
            if done:
                if episode_reward > prev_ep_rewards:
                    prev_ep_rewards = episode_reward
                    print("improvement ", episode_reward, episode)

                # model.theta[a] += ALPHA*(r - value_t1) * model.activation_gradient(Q_t1)[a,a] * model.gradient(s_t1)
            else:
                # Back-propagate   
                e_softmax = (r + GAMMA*max_value_t2 - value_t1) * model.sigmoid_gradient(Q_t1[a])
                model.w_2_3[:,a] += ALPHA * e_softmax * model.l_2
                model.w_2_3_b[a] += ALPHA * e_softmax

                error_l_2 = e_softmax * model.w_2_3[:,a]
                error_l_2 = error_l_2 * model.sigmoid_gradient(model.l_2)
                model.w_1_2 += ALPHA * (error_l_2.reshape(HIDDEN_SIZE,1).dot(model.input_gradient(s_t1).reshape(INPUT_SIZE,1).T)).T
                model.w_1_2_b += ALPHA * error_l_2



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

            display[2][0] = ((model.w_2_3[:,2].sum() - theta_sum.min())*255 / (theta_sum.max() - theta_sum.min()), 0, 0)
            display[2][1] = (0, (model.w_2_3[:,0].sum() - theta_sum.min())*255 / (theta_sum.max() - theta_sum.min()), 0)
            display[2][2] = (0, 0, (model.w_2_3[:,1].sum() - theta_sum.min())*255 / (theta_sum.max() - theta_sum.min()))

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