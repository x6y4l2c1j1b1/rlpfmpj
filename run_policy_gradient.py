"""
Policy Gradient, Reinforcement Learning.

The cart pole example

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

from RL_brain_policy_gradient import PolicyGradient
from env_new_high_a import Env
import numpy as np
import matplotlib.pyplot as plt

# DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold
# RENDER = False  # rendering wastes time

# env = gym.make('CartPole-v0')
# env.seed(1)     # reproducible, general Policy gradient has high variance
# env = env.unwrapped

# print(env.action_space)
# print(env.observation_space)
# print(env.observation_space.high)
# print(env.observation_space.low)

n_actions  = len(np.arange(-0.25,0.25,0.01))
n_features = 8

def run_stock():
    total_episode = 100
    win = 0
    lose = 0
    profit_estimation = 0
    for i_episode in range(total_episode):

        env.reset(np.random.randint(0,10), np.random.randint(0,10))
        observation = np.array(env.state)

        while True:

            action = RL.choose_action(observation)

            observation_, reward, done= env.take_action(action)
            observation_ = np.array(observation_)

            RL.store_transition(observation, action, reward)

            if done:
                ep_rs_sum = sum(RL.ep_rs)

                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
                # if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering
                print("episode:", i_episode, "  reward:", int(running_reward))

                vt = RL.learn()

                # if i_episode == 0:
                #     plt.plot(vt)    # plot the episode vt
                #     plt.xlabel('episode steps')
                #     plt.ylabel('normalized state-action value')
                #     plt.show()
                break

            observation = observation_

        if observation[0] >= 1000000:
            win += 1
        else:
            lose += 1

        profit_estimation = (profit_estimation * i_episode + (observation[0] - 1000000.))/(i_episode+1.)
        print('total_value: %d, profit_estimation: %d' % (observation[0], profit_estimation))
    print('train, win: %d, lose: %d, profit_estimation: %d' % (win, lose, profit_estimation))

def test():
    win = 0
    lose = 0
    total_episode = 100
    profit_estimation = 0
    for i_episode in range(total_episode):

        env.reset(np.random.randint(0,10), np.random.randint(0,10))
        observation = np.array(env.state)

        while True:

            action = RL.choose_action(observation)

            observation_, reward, done= env.take_action(action)
            observation_ = np.array(observation_)

            RL.store_transition(observation, action, reward)

            if done:
                break

            observation = observation_

        if observation[0] >= 1000000:
            win += 1
        else:
            lose += 1

        profit_estimation = (profit_estimation * i_episode + (observation[0] - 1000000.))/(i_episode+1.)
        print('total_value: %d, profit_estimation: %d' % (observation[0], profit_estimation))
    print('test, win: %d, lose: %d, profit_estimation: %d' % (win, lose, profit_estimation))

if __name__ == '__main__':
    env = Env()
    RL = PolicyGradient(
        n_actions=n_actions,
        n_features=n_features,
        learning_rate=0.02,
        reward_decay=0.99,
        # output_graph=True,
    )
    run_stock()
    test()
