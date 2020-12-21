import gym
from gym import error, spaces, utils
from gym.utils import seeding

from .PybulletClient import *
from .PybulletInterface import *

import numpy
import os


class FoodGatheringAdvancedMultiEnv(gym.Env, PybulletInterface):
    metadata = {'render.modes': ['human']}

    def __init__(self, render = False):
        gym.Env.__init__(self)

        self.robots_count = 16
        self.lidar_points = 64
        PybulletInterface.__init__(self, render = render, lidar_points = self.lidar_points, world_size = 1, view_camera_distance = 8.5, view_camera_angle = -89.9)

        self.action_space       = spaces.Box(low=-1.0, high=1.0, shape=(self.robots_count*2,), dtype=numpy.float32)
        self.observation_space  = spaces.Box(low=-1.0, high=1.0, shape=(self.robots_count, 6, self.lidar_points), dtype=numpy.float32)

        
    def step(self, action):
        self.step_interface()

        
        done    = False
        reward = 0
        
        for i in range(self.robots_count):
            vl = 50.0*numpy.clip(action[2*i + 0], -1.0, 1.0)
            vr = 50.0*numpy.clip(action[2*i + 1], -1.0, 1.0)

            self.robots[i].set_velocity(vl, vr)        

            distance = self.closest_food_distance(i)
            reward+= 0.001*numpy.exp(-distance)

        reward = reward/self.robots_count
       

        for i in range(self.robots_count):
            food_id = self.on_food(i)

            if food_id != -1:
                reward+= 1.0
                self.pb_client.removeBody(self.foods[food_id])
                del self.foods[food_id]
                if len(self.foods) == 0:
                    done = True
            elif self.on_fragile(i):
                reward = -0.1
            elif self.on_hazard(i):
                reward = -1.0
                done   = True
            elif self.out_board(i):
                reward = -1.0
                done   = True


        if self.steps >= 1000:
            reward = -1.0
            done   = True

        
        for i in range(4):
            self.pb_client.stepSimulation()

        return self._update_observation(), reward, done, None

        
    
    def reset(self):
        robots_count    = 1*self.robots_count
        targets_count   = 0
        hazards_count   = 8*self.robots_count
        obstacles_count = 4*self.robots_count
        fragile_count   = 8*self.robots_count
        moving_count    = 0
        foods_count     = 20*self.robots_count
        
        self.reset_interface(targets_count, robots_count, hazards_count, obstacles_count, fragile_count,  moving_count, foods_count)

        return self._update_observation()
        
    def render(self):
        pass

    def close(self):
        pass

    def _update_observation(self):
        result    = numpy.zeros((self.robots_count, 6, self.lidar_points), dtype=numpy.float32)
        for i in range(self.robots_count):
            result[i] = self._update_observation_robot(i, self.lidar_points)

        return result


    def _update_observation_robot(self, robot_id, lidar_points):
        lidar   = self.get_lidar(robot_id)

        vl, vr  = self.robots[robot_id].get_wheel_velocity()

        result    = numpy.zeros((6, lidar_points), dtype=numpy.float32)

        result[0] = numpy.tanh(vl*numpy.ones(lidar_points)/50.0) #robot velocity, squeezed by tanh
        result[1] = numpy.tanh(vr*numpy.ones(lidar_points)/50.0)
        result[2] = lidar[3]        #obstacles lidar
        result[3] = lidar[2]        #hazards lidar
        result[4] = lidar[4]        #fragiles lidar
        result[5] = lidar[6]        #food lidar
      
        #self.render_lidar(lidar)

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
 