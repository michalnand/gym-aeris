import gym
from .GridInterface     import *

import numpy
import os


class GridTargetSearchAEnv(gym.Env, GridInterface):
    def __init__(self, render = False, view_camera_distance = 1.5, view_camera_angle = -80.0):
        gym.Env.__init__(self) 
        GridInterface.__init__(self, render, view_camera_distance, view_camera_angle) 
        
        self.grid_map = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0],
            [0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0],
            [0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]

        self.reset_interface(self.grid_map)
        obs = self.update_observation()

        self.action_space       = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=numpy.float32)
        self.observation_space  = gym.spaces.Box(low=-1.0, high=1.0, shape=obs.shape, dtype=numpy.float32)

        
    def step(self, action):
        self.step_interface(action)

        reward  = 0.0
        done    = False

        if self.steps >= 1000:
            reward = 0.0
            done   = True
        
        elif self.on_target(0, 0):
            reward = 1.0
            done   = True 

        elif self.out_board(0):
            reward = -1.0
            done   = True

        return self.update_observation(), reward, done, None

    def reset(self):
        self.reset_interface(self.grid_map)
        return self.update_observation()
        
    def render(self):
        pass

    def close(self):
        pass

  

class GridTargetSearchADiscreteEnv(gym.Env):
    def __init__(self, render = False, view_camera_distance = 1.5, view_camera_angle = -80.0):
        gym.Env.__init__(self) 
        
        self.env = GridTargetSearchAEnv(render, view_camera_distance, view_camera_angle)
       
        self.action_space       = gym.spaces.Discrete(16)
        self.observation_space  = self.env.observation_space

        actions = []

        actions.append([ 0.0,  0.0])
        actions.append([ 0.0,  0.2])
        actions.append([ 0.2,  0.0]) 
        actions.append([ 0.0, -0.2])
        actions.append([-0.2,  0.0])
        actions.append([ 0.0,  0.5])
        actions.append([ 0.5,  0.0])
        actions.append([ 0.0, -0.5])
        actions.append([-0.5,  0.0])
        actions.append([ 0.0,  1.0])
        actions.append([ 1.0,  0.0])
        actions.append([ 1.0,  1.0])
        actions.append([ 0.5,  -0.5])
        actions.append([-0.5,  0.5])
        actions.append([-0.2,  -0.2])
        actions.append([-0.5,  -0.5])

        self.actions = numpy.array(actions)


    def step(self, action):
        return self.env.step(self.actions[action])

    def reset(self):
        return self.env.reset()
        
    def render(self):
        pass

    def close(self):
        pass