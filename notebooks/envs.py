import utils
import gym

taxi = utils.EnvWrapper(gym.envs.make("Taxi-v3"))
blackjack = utils.EnvWrapper(gym.envs.make("Blackjack-v1"))
frozenlake = utils.EnvWrapper(gym.envs.make("FrozenLake-v1"))