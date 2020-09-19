import gym
import numpy as np
import sys
import math
import time

# np.set_printoptions(threshold=sys.maxsize)

# env = gym.make("MountainCar-v0")
env = gym.make("Acrobot-v1")

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EXPLORE = False

EPISODES = 25000
SHOW_EVERY = 100

# Exploration settings
epsilon = 1  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES#//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

def get_discrete(state):
    discrete_state = state/20
    # print(discrete_state)
    # sys.exit()
    return discrete_state

episode_reward = 0
prev_ep_rewards = -200
rewards = []
q_table = {}
states_counter = 0
for episode in range(EPISODES):
    learn_rate = max(0.1, LEARNING_RATE-(episode**(math.log(EPISODES, 1000))/EPISODES))
    state = str(get_discrete(env.reset()))
    if state not in q_table:
        states_counter += 1
        q_table[state] = np.random.uniform(low=0, high=3, size=env.action_space.n)
    done = False

    # print(states_counter)
    if episode and episode % SHOW_EVERY == 0:
        render = True
        print(episode, int(np.mean(rewards[-SHOW_EVERY:])), int(np.std(rewards[-SHOW_EVERY:])), int(np.min(rewards[-SHOW_EVERY:])), int(np.max(rewards[-SHOW_EVERY:])), "qValues", states_counter)
    else:
        render = False

    rewards.append(episode_reward)
    episode_reward = 0
    closest = -10
    step_counter = 0
    while not done:
        time.sleep(1)
        step_counter += 1

        ###############
        # Take an action
        ###############

        if EXPLORE:
            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(q_table[state])
            else:
                # Get random action
                action = np.random.randint(0, env.action_space.n)
        else:
            action = np.argmax(q_table[state])
            

        ###############
        # Update Q Value
        ###############

        new_state, reward, done, _ = env.step(action)
        # if reward != 0:
            # print("reward", reward)
        if new_state[0] > closest:
        	closest = new_state[0]
        episode_reward += reward
        new_state = str(get_discrete(new_state))
        if new_state not in q_table:
            states_counter += 1
            q_table[new_state] = np.random.uniform(low=0, high=3, size=env.action_space.n)
        # else:
            # print("seen ", episode)

        if episode % SHOW_EVERY == 0:
            env.render()

        # If simulation did not end yet after last step - update Q table
        if not done:

            # Maximum possible Q value in next step (for new state)
            max_future_q = np.max(q_table[new_state])

            # Current Q value (for current state and performed action)
            current_q = q_table[state][action]

            # And here's our equation for a new Q value for current state and action
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            # Update Q table with new Q value
            q_table[state, action] = new_q


        # Simulation ended (for any reson) - if goal position is achived - update Q value with reward directly
        # elif new_state[0] >= env.goal_position and is_learn:
        else:
            if episode_reward > prev_ep_rewards:
                prev_ep_rewards = episode_reward
                print("improvement ", episode_reward, episode)
                # np.save("qtables/qtable.npy", q_table)
            #q_table[discrete_state + (action,)] = reward
            # q_table[state, action] = 0 
        # else:
        #     q_table[discrete_state + (action,)] = (closest + env.observation_space.low[0])


        state = new_state

    # Decaying is being done every episode if episode number is within decaying range
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value


env.close()