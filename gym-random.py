import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import regularizers
from keras import optimizers
import keras.utils 

import gym

render = False
env = gym.make('MountainCar-v0')
print('action space:', env.action_space)
reward = None
rewards_list = []
for i_episode in range(1000):
    if i_episode %1 == 0:
        print('episode:', i_episode)
    total_reward = 0
    observation = env.reset()
    for t in range(5000):
        if render:
            env.render()
        print('obs, shape', observation.shape)
        print(observation)
        if reward != 0:
            print(reward)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            print("Total reward accrued:", total_reward)
            rewards_list.append(total_reward)
            break

print("average reward accrued:", sum(rewards_list) / len(rewards_list))
