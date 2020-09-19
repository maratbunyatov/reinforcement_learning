import gym
import numpy as np
import sys
import math
import time
import ast

# np.set_printoptions(threshold=sys.maxsize)

# env = gym.make("MountainCar-v0")
env = gym.make("Breakout-ram-v0")

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EXPLORE = False

EPISODES = 25000
SHOW_EVERY = 100
SLEEP = .001
PRINT = False
Y = [0,16]
OBS = [0,1,3]

MEM_ADDR = [72,99,101] #[70,99,] #[70,72,90,99,101,105]

# Exploration settings
epsilon = 1  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES#//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

def get_discrete(state):
    discrete_state = state//[8,8,16]
    return discrete_state

episode_reward = 0
prev_ep_rewards = -20000
rewards = []
q_table = {}
states_counter = 0
left_right = 0
for episode in range(EPISODES):
    learn_rate = max(0.1, LEARNING_RATE-(episode**(math.log(EPISODES, 1000))/EPISODES))
    state_continuos = env.reset()[MEM_ADDR]
    state = np.append(get_discrete(state_continuos), [left_right])
    # state = str(env.reset()[[70,72,90,99,101,105]])
    if str(state[OBS]) not in q_table and Y[0] < state[2] < Y[1]:
        states_counter += 1
        q_table[str(state[OBS])] = np.random.uniform(low=0, high=2, size=env.action_space.n-1)
    done = False

    # print(states_counter)
    if episode and episode % SHOW_EVERY == 0:
        render = True
        print(episode, np.mean(rewards[-SHOW_EVERY:]), f"{np.std(rewards[-SHOW_EVERY:]):.2f}", int(np.min(rewards[-SHOW_EVERY:])), int(np.max(rewards[-SHOW_EVERY:])), f"(qValues {states_counter})")
    else:
        render = False

    rewards.append(episode_reward)
    episode_reward = 0
    step_counter = 0
    up_down = 0
    while not done:
        step_counter += 1

        ###############
        # Take an action
        ###############
        if EXPLORE:
            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(q_table[str(state[OBS])])
            else:
                # Get random action
                action = np.random.randint(0, env.action_space.n)
        else:
            if Y[0] < state[2] < Y[1]:
                action = np.argmax(q_table[str(state[OBS])])
            else:
               action = 0
        # if state[2] < 8:
        #     action = 1
            

        ###############
        # Update Q Value
        ###############

        new_state_continuos, hit, done, _ = env.step(action+1)
        new_state_continuos = new_state_continuos[MEM_ADDR]

        if state_continuos[1] < new_state_continuos[1]:
            left_right = 1
        else:
            left_right = 0  
        # left_right = state_continuos[1] - new_state_continuos[1]

        new_state = np.append(get_discrete(new_state_continuos), [left_right])
        reward = -abs((new_state[0] - new_state[1])**2)
        episode_reward += hit
        if PRINT: print(new_state, reward, action)

        if str(new_state[OBS]) not in q_table and Y[0] < new_state[2] < Y[1]:
            states_counter += 1
            q_table[str(new_state[OBS])] = np.random.uniform(low=0, high=2, size=env.action_space.n-1)
        # else:
            # print("seen ", episode)

        if episode % SHOW_EVERY == 0:
            time.sleep(SLEEP*max(1, prev_ep_rewards))
            env.render()

        # If simulation did not end yet after last step - update Q table
        if not done and Y[0] < new_state[2] < Y[1] and Y[0] < state[2] < Y[1]:

            # Maximum possible Q value in next step (for new state)
            max_future_q = np.max(q_table[str(new_state[OBS])])

            # Current Q value (for current state and performed action)
            current_q = q_table[str(state[OBS])][action]

            # And here's our equation for a new Q value for current state and action
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            # Update Q table with new Q value
            if Y[0] < new_state[2] < Y[1]:
                q_table[str(state[OBS])][action] = new_q


        # Simulation ended (for any reson) - if goal position is achived - update Q value with reward directly
        # elif new_state[0] >= env.goal_position and is_learn:
        else:
            if episode_reward > prev_ep_rewards:
                prev_ep_rewards = episode_reward
                print("improvement ", episode_reward, episode)

        state = new_state
        state_continuos = new_state_continuos

    # Decaying is being done every episode if episode number is within decaying range
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value


env.close()