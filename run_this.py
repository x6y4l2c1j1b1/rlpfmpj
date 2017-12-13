from RL_brain import DeepQNetwork
from env_new import Env
import numpy as np


def run_stock():
    step = 0
    win = 0
    lose = 0
    for episode in range(100):
        # initial observation
        print episode
        env.reset(np.random.randint(0,10), np.random.randint(0,10))
        #observation = env.reset()
        observation = np.array(env.state)
        while True:
            # fresh env
            #env.render()

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

    # end of game
        print(observation)
    print('win: %d lose: %d' % (win, lose))
    #env.destroy()


if __name__ == "__main__":
    # maze game
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
    #env.after(100, run_maze)
    #env.mainloop()
    RL.plot_cost()