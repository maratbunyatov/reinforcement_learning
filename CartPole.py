import gym
import numpy as np
import sys
import math

# np.set_printoptions(threshold=sys.maxsize)

# env = gym.make("MountainCar-v0")
env = gym.make("CartPole-v1")

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EXPLORE = True

EPISODES = 25000
SHOW_EVERY = 1000

DISCRETE_OS_SIZE = [20, 20, 20, 20]
NEW_RANGE = 5
env.observation_space.low[1] = -NEW_RANGE
env.observation_space.low[3] = -NEW_RANGE
env.observation_space.high[1] = NEW_RANGE
env.observation_space.high[3] = NEW_RANGE
print(env.observation_space.high, env.observation_space.low)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

# Exploration settings
epsilon = 1  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES#//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

is_learn = True

try:
	q_table = np.load(f"qtables/qtable.npy")
	is_learn = False
except:
	q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
    # print(f"state {state}")
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    # print(f"discr {discrete_state}")
    return tuple(discrete_state.astype(np.int))  # we use this tuple to look up the 3 Q values for the available actions in the q-table

episode_reward = 0
prev_ep_rewards = -200
rewards = []
for episode in range(EPISODES):
    learn_rate = max(0.1, LEARNING_RATE-(episode**(math.log(EPISODES, 1000))/EPISODES))
    discrete_state = get_discrete_state(env.reset())
    done = False

    if episode and episode % SHOW_EVERY == 0:
        render = True
        print(episode, int(np.mean(rewards[-SHOW_EVERY:])), int(np.std(rewards[-SHOW_EVERY:])), int(np.min(rewards[-SHOW_EVERY:])), int(np.max(rewards[-SHOW_EVERY:])))
    else:
        render = False

    rewards.append(episode_reward)
    episode_reward = 0
    closest = -10
    step_counter = 0
    while not done:
        step_counter += 1
        if EXPLORE:
            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(q_table[discrete_state])
            else:
                # Get random action
                action = np.random.randint(0, env.action_space.n)
        else:
            action = np.argmax(q_table[discrete_state])

        new_state, reward, done, _ = env.step(action)
        if new_state[0] > closest:
        	closest = new_state[0]
        episode_reward += reward

        new_discrete_state = get_discrete_state(new_state)

        if episode % SHOW_EVERY == 0:
            env.render()
        #new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        # If simulation did not end yet after last step - update Q table
        if not done:

            # Maximum possible Q value in next step (for new state)
            max_future_q = np.max(q_table[new_discrete_state])

            # Current Q value (for current state and performed action)
            current_q = q_table[discrete_state + (action,)]

            # And here's our equation for a new Q value for current state and action
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            # Update Q table with new Q value
            if is_learn:
                q_table[discrete_state + (action,)] = new_q


        # Simulation ended (for any reson) - if goal position is achived - update Q value with reward directly
        # elif new_state[0] >= env.goal_position and is_learn:
        else:
            if episode_reward > prev_ep_rewards:
                prev_ep_rewards = episode_reward
                print(episode_reward, episode)
                # np.save("qtables/qtable.npy", q_table)
            #q_table[discrete_state + (action,)] = reward
            q_table[discrete_state + (action,)] = 0 
        # else:
        #     q_table[discrete_state + (action,)] = (closest + env.observation_space.low[0])


        discrete_state = new_discrete_state

    # Decaying is being done every episode if episode number is within decaying range
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value


env.close()