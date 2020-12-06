import sys
sys.path.insert(0,'..')

import gym_aeris.envs

env = gym_aeris.envs.GoalEnv()
env.reset()

while True:
    state, reward, done, info = env.step(0)
    if done:
        env.reset()