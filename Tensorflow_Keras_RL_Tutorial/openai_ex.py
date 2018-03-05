import gym
import numpy as np
import tflearn
from tflearn.layers.core import input_data
from tflearn.layers.core import dropout
from tflearn.layers.core import fully_connected
from tflearn.layers.estimator import regression
from collections import Counter


LR = 1e-3
env = gym.make('CartPole-v1')
env.reset()
goal_steps = 500
score_requirement = 50
inital_games = 10000


def some_random_games_first():
    for episode in range(20):
        env.reset()
        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break


# some_random_games_first()

