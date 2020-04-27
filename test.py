import gym
import argparse
from math import log2
import time
from random import randint

import matplotlib.pyplot as plt


class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self,):
        return self.action_space.sample()


def main():
    """

    :return:
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--display', action='store_true')
    parser.add_argument('game', nargs="?", default="Pong-v0")
    args = parser.parse_args()

    env = gym.make(args.game)
    # env.seed(0)
    num_episodes = 1  # 20
    num_maxstep = 1000  # 100

    agent = RandomAgent(env.action_space)

    reward = 0
    done = False
    # print(dir(env.ale))

    ram_list = []
    state_list = []

    for i_episode in range(num_episodes):
        observation = env.reset()
        o_ram = env.ale.getRAM()
        index_dict = {}
        for t in range(num_maxstep):
            env.render()
            observation, reward, done, info = env.step(agent.act())
            # print("reward", reward)
            # print("observation.shape", observation.shape)

            # Clone the system state as a int type
            # clone_state = env.clone_state()
            # print(clone_state, type(clone_state), clone_state.shape)

            # cloned system state index from 163 to 674 represents ram state
            # for i in range(163, 675):
            #     perturbation = randint(-1, 1)
            #     clone_state[i] += perturbation

            # print(clone_state, type(clone_state), clone_state.shape)

            # env.restore_state(clone_state)

            # time.sleep(0.5)

            # image = env.ale.getScreenRGB2()
            # print("image", image, type(image), image.shape)
            #
            n_ram = env.ale.getRAM()
            # print(n_ram)

            for i in range(128):
                if n_ram[i] != o_ram[i]:
                    # print(i, ":", o_ram[i], "->", n_ram[i])
                    if i in index_dict:
                        index_dict[i].append(n_ram[i])
                    else:
                        index_dict[i] = [o_ram[i], n_ram[i]]

            o_ram = n_ram

            # Encode the cloned state to a numpy array
            # encode_clone_state = env.ale.encodeState(clone_state)
            # print(encode_clone_state, type(encode_clone_state), encode_clone_state.size)
            #
            # # Decode the numpy array to the int type state
            # original_state = env.ale.decodeState(encode_clone_state)
            # print(original_state, type(original_state), log2(original_state))
            #
            # # Restore the system state from the int type state
            # env.ale.restoreSystemState(original_state)
        print(index_dict)

        for item in index_dict:
            plt.plot(range(len(index_dict[item])), index_dict[item])
            plt.title("RAM INDEX:" + str(item))
            plt.show()

    env.close()
    return None


main()
