import sys
sys.path.insert(0,'..')

import gym_aeris.envs
from AgentRandom import *


env = gym_aeris.envs.GoalEnv()
env.reset()

agent = AgentRandom(env)

while True:
    agent.main()