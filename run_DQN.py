from RL_brain_DQN import DeepQNetwork
from env_new import Env
import numpy as np


def run_stock():
    step = 0
    win = 0
    lose = 0
    total_episode = 1000
    profit_estimation = 0
    for episode in range(total_episode):
        # initial observation
        print(episode)
        env.reset(np.random.randint(0,10), np.random.randint(0,10))
        observation = np.array(env.state)
        while True:

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.take_action(action)
            observation_ = np.array(observation_)
            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1
        if observation[0] >= 1000000:
            win += 1
        else:
            lose += 1

        profit_estimation = (profit_estimation * episode + (observation[0] - 1000000.))/(episode+1.)

    # end of game
        print('total_value: %d, profit_estimation: %d' % (observation[0], profit_estimation))
    RL.save_model("test001")
    print('train, win: %d, lose: %d, profit_estimation: %d' % (win, lose, profit_estimation))
    #env.destroy()

def test_stock():
    step = 0
    win = 0
    lose = 0
    total_episode = 1000
    profit_estimation = 0

    profits = []
    actions = [0,0,0,0,0,0,0]

    for episode in range(total_episode):
        # initial observation
        print(episode)
        env.reset(np.random.randint(0,10), np.random.randint(0,10))
        observation = np.array(env.state)
        while True:

            # RL choose action based on observation
            action = RL.choose_greedy_action(observation)
            #count actions
            actions[action] = actions[action] + 1
            # RL take action and get next observation and reward
            observation_, reward, done = env.take_action(action)
            observation_ = np.array(observation_)
            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1
        if observation[0] >= 1000000:
            win += 1
        else:
            lose += 1
        profits.append(observation[0] - 1000000.0)
        profit_estimation = (profit_estimation * episode + (observation[0] - 1000000.))/(episode+1.)

    # end of game
        print('total_value: %d, profit_estimation: %d' % (observation[0], profit_estimation))
    #print("test_profits")
    #print(np.array(profits))
    #print("test_actions")
    #print(np.array(actions))

    np.savetxt("profits_test.csv",np.array(profits),delimiter=",")
    np.savetxt("actions_test.csv",np.array(actions),delimiter=",")
    print('test, win: %d, lose: %d, profit_estimation: %d' % (win, lose, profit_estimation))

def run_random():
    step = 0
    win = 0
    lose = 0
    total_episode = 1000
    profit_estimation = 0

    profits = []
    actions = [0,0,0,0,0,0,0]

    for episode in range(total_episode):
        # initial observation
        print(episode)
        env.reset(np.random.randint(0,10), np.random.randint(0,10))
        observation = np.array(env.state)
        while True:
            # fresh env

            # RL choose action based on observation
            action = np.random.randint(0,7)
            #count actions
            actions[action] = actions[action] + 1
            # RL take action and get next observation and reward
            observation_, reward, done = env.take_action(action)
            observation_ = np.array(observation_)
            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1
        if observation[0] >= 1000000:
            win += 1
        else:
            lose += 1
        profits.append(observation[0] - 1000000.0)
        profit_estimation = (profit_estimation * episode + (observation[0] - 1000000.))/(episode+1.)
        print('total_value: %d, profit_estimation: %d' % (observation[0], profit_estimation))

    np.savetxt("profits_random.csv",np.array(profits),delimiter=",")
    np.savetxt("actions_random.csv",np.array(actions),delimiter=",")
    print('random, win: %d, lose: %d, profit_estimation: %d' % (win, lose, profit_estimation))


if __name__ == "__main__":
    env = Env()
    n_actions = 7
    n_features = 8
    RL = DeepQNetwork(n_actions, n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    
    run_stock()
    test_stock()
    run_random()
    RL.plot_cost()
