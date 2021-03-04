import sys
sys.path.insert(0,'..')

import time

import gym_aeris.envs
from AgentRandomMulti import *


env = gym_aeris.envs.SwarmFoodGatheringEnv()
env.reset()

agent = AgentRandomMulti(env)

k = 0.998
fps = 0

while True:

    time_start = time.time()
    agent.main(verbose=False)
    time_stop  = time.time() 

    fps = (1.0 - k)*fps + k*1.0/(time_stop - time_start + 0.00001)

    #env.render(0)
    env.render(3)

    print("fps = ", round(fps, 2))