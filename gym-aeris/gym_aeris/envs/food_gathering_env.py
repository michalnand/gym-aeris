import gym
from gym import error, spaces, utils
from gym.utils import seeding

from .PybulletClient import *
from .PybulletInterface import *

import numpy
import os


class FoodGatheringEnv(gym.Env, PybulletInterface):
    metadata = {'render.modes': ['human']}

    def __init__(self, render = False):
        gym.Env.__init__(self)

        self.lidar_points = 32
        PybulletInterface.__init__(self, render = render, lidar_points = self.lidar_points)

        self.action_space       = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=numpy.float32)
        self.observation_space  = spaces.Box(low=-1.0, high=1.0, shape=(3, self.lidar_points), dtype=numpy.float32)

        
    def step(self, action):
        self.step_interface()
        
        vl = 50.0*numpy.clip(action[0], -1.0, 1.0)
        vr = 50.0*numpy.clip(action[1], -1.0, 1.0)

        self.robots[0].set_velocity(vl, vr)
        
        distance = self.closest_food_distance()
        reward = 0.001*numpy.exp(-distance)
        
        done    = False

        food_id = self.on_food(0)

        if food_id != -1:
            reward = 1.0
            self.pb_client.removeBody(self.foods[food_id])
            del self.foods[food_id]
            if len(self.foods) == 0:
                done = True
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
        targets_count   = 0
        hazards_count   = 0
        obstacles_count = 1
        fragile_count   = 0
        moving_count    = 0
        foods_count     = 10
        
        self.reset_interface(targets_count, robots_count, hazards_count, obstacles_count, fragile_count,  moving_count, foods_count)

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
        result[3] = lidar[6]        #food lidar
      
        return result

    def _dummy_follow(self):
        items_r, items_yaw = self.get_items_relative_position(self.robots[0].pb_robot, self.foods)

        if numpy.abs(items_yaw[0]) > 0.3:
            if items_yaw[0] > 0.0:
                self.robots[0].set_velocity(-5.0, 5.0)
            else:
                self.robots[0].set_velocity(5.0, -5.0)
        else:
            self.robots[0].set_velocity(50.0, 50.0)
 

    