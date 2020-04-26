import gym
import argparse
from math import log2


class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
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
    num_episodes = 1 # 20
    num_maxstep = 1 # 100

    agent = RandomAgent(env.action_space)

    reward = 0
    done = False
    print(dir(env.ale))


    for i_episode in range(num_episodes):
        observation = env.reset()
        for t in range(num_maxstep):
            # env.render()
            action = agent.act(observation, reward, done)
            observation, reward, done, info = env.step(action)
            print("observation.shape", observation.shape)

            # Clone the system state as a int type
            clone_state = env.ale.cloneSystemState()

            print("RAM", env.ale.getRAM(), env.ale.getRAM().shape)
            # env.ale.loadROM(env.ale.getRAM())

            # Encode the cloned state to a numpy array
            encode_clone_state = env.ale.encodeState(clone_state)
            print(encode_clone_state, type(encode_clone_state), encode_clone_state.size)

            # Decode the numpy array to the int type state
            original_state = env.ale.decodeState(encode_clone_state)
            print(original_state, type(original_state), log2(original_state))

            # Restore the system state from the int type state
            env.ale.restoreSystemState(original_state)

            # observation, reward, done, info = env.step(action)

    env.close()
    return None


main()
