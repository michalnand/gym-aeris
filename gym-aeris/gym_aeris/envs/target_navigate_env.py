import gym
from gym import error, spaces, utils
from gym.utils import seeding

from .PybulletClient import *
from .PybulletInterface import *

import numpy
import os


class TargetNavigateEnv(gym.Env, PybulletInterface):
    metadata = {'render.modes': ['human']}

    def __init__(self, render = False):
        gym.Env.__init__(self)

        self.lidar_points = 64
        PybulletInterface.__init__(self, render = render, lidar_points = self.lidar_points)

        self.action_space       = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=numpy.float32)
        self.observation_space  = spaces.Box(low=-1.0, high=1.0, shape=(4, self.lidar_points), dtype=numpy.float32)

        
    def step(self, action):
        self.step_interface()


        vl = 50.0*numpy.clip(action[0], -1.0, 1.0)
        vr = 50.0*numpy.clip(action[1], -1.0, 1.0)

        self.robots[0].set_velocity(vl, vr)
        
        '''
        self._dummy_follow()  
        self.render_lidar(self.lidar)
        '''

        distance = self.target_distance()
        reward = 0.01*numpy.exp(-distance)
        
        done    = False

        if self.on_target(0, 0):
            reward = 1.0
            done   = True 
        elif self.out_board(0):
            reward = -1.0
            done   = True
        elif self.steps >= 1000:
            reward = -1.0
            done   = True

        for i in range(4):
            self.pb_client.stepSimulation()

        return self._update_observation(robot_id=0, lidar_points=self.lidar_points), reward, done, None

        
    
    def reset(self):
        robots_count    = 1 
        targets_count   = 1
        hazards_count   = 0
        obstacles_count = 1
        fragile_count   = 0
        moving_count    = 0
        foods_count     = 0
        
        self.reset_interface(targets_count, robots_count, hazards_count, obstacles_count, fragile_count,  moving_count, foods_count  )

        return self._update_observation(robot_id=0, lidar_points=self.lidar_points)
        
    def render(self):
        pass

    def close(self):
        pass

    def _update_observation(self, robot_id, lidar_points):
        lidar   = self.get_lidar(robot_id)

        vl, vr  = self.robots[robot_id].get_wheel_velocity()

        result    = numpy.zeros((4, lidar_points), dtype=numpy.float32)

        result[0] = numpy.tanh(vl*numpy.ones(lidar_points)/50.0) #robot velocity, squeezed by tanh
        result[1] = numpy.tanh(vr*numpy.ones(lidar_points)/50.0)
        result[2] = lidar[3]        #obstacles lidar
        result[3] = lidar[1]        #target lidar

        return result

    def _dummy_follow(self):

        obs = self._update_observation(robot_id=0, lidar_points=self.lidar_points)

        target = obs[3]

        target_idx = numpy.argmax(target)

        if target_idx >= self.lidar_points//2:
            self.robots[0].set_velocity(-5.0 + 3.0, 5.0 + 3.0)
        else:
            self.robots[0].set_velocity(5.0 + 3.0, -5.0 + 3.0)

       