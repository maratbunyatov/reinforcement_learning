import numpy as np
import sys
import time
from PIL import Image
import cv2

np.set_printoptions(formatter={'float': '{: 0.2f}'.format}, suppress=True)

ALPHA = 0.1

EPISODES = 2_000
SHOW_EVERY = 100

HIDDEN_SIZE = 10
INPUT_SIZE = 2
OUTPUT_SIZE = 2

class Model:
  def __init__(self):
    self.l_2 = np.random.randn(HIDDEN_SIZE) / np.sqrt(HIDDEN_SIZE)

    self.w_1_2 = np.random.randn(INPUT_SIZE, HIDDEN_SIZE) / np.sqrt(INPUT_SIZE) # Xavier intitialization 
    self.w_1_2_b = np.random.randn(HIDDEN_SIZE) / np.sqrt(INPUT_SIZE) # bias weights
    self.w_2_3 = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) / np.sqrt(HIDDEN_SIZE)
    self.w_2_3_b = np.random.randn(OUTPUT_SIZE) / np.sqrt(HIDDEN_SIZE) 

  def perceive(self, s):
    # forward propagate
    self.l_2 = self.sigmoid_activation(self.w_1_2.T.dot(s) + self.w_1_2_b)
    l3_activations = self.softmax_activation(self.w_2_3.T.dot(self.l_2) + self.w_2_3_b)
    return l3_activations

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
    # Sigmoid. Assumes s is the sigmoid value
    return s * (1 - s)


# initialize model
model = Model()

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

    if episode and not episode % SHOW_EVERY:
        render = True
        print(episode, np.mean(errors[-SHOW_EVERY:]), f"{np.std(errors[-SHOW_EVERY:]):.2f}", int(np.sum(errors[-SHOW_EVERY:])), 
            # f" ...  (theta: \n{model.theta[:,[PADDLE_IDX,X_IDX,Y_IDX]]})"
            # f" ...  (w_1_2: \n{model.w_1_2})\n (w_2_3: \n{model.w_2_3})", 
            # f"\nQ: {Q_t1:.2f} Weights: [{theta_sum[2]:.2f}, {theta_sum[0]:.2f}, {theta_sum[1]:.2f}]"
        )
        errors = []

    # Peceive
    Q_t1 = model.perceive(s_t1)
    # print(f"Q: [{Q_t1[2]:.2f}, {Q_t1[0]:.2f}, {Q_t1[1]:.2f}] Weights: [{theta_sum[2]:.2f}, {theta_sum[0]:.2f}, {theta_sum[1]:.2f}]")

    # Learn
    e = answers[str(s_t1)] - Q_t1
    # print(f"episode: {episode}, Q_t1: {Q_t1}, target: {answers[str(s_t1)]}, error: {e}")
    errors.append(abs(e))
    if episode > EPISODES-10:
        print(f"s_t1:{s_t1}, Q_t1:{Q_t1}, target:{answers[str(s_t1)]}, error:{e}")

    # Back-propagate  
    e_softmax = e.dot(model.softmax_gradient(Q_t1)).reshape(OUTPUT_SIZE, 1)
    model.w_2_3 += ALPHA * e_softmax.dot(model.l_2.reshape(HIDDEN_SIZE, 1).T).T
    model.w_2_3_b += ALPHA * e_softmax.reshape(OUTPUT_SIZE,)

    error_l_2 = e_softmax.T.dot(model.w_2_3.T).reshape(HIDDEN_SIZE, 1)
    error_l_2 = error_l_2.reshape(HIDDEN_SIZE,) * model.sigmoid_gradient(model.l_2)
    model.w_1_2 += ALPHA * (error_l_2.reshape(HIDDEN_SIZE, 1).dot(s_t1.reshape(INPUT_SIZE, 1).T)).T
    model.w_1_2_b += ALPHA * error_l_2



    