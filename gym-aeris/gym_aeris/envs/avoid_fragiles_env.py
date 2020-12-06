import gym
from gym import error, spaces, utils
from gym.utils import seeding

from .PybulletClient import *
from .PybulletInterface import *

import numpy
import os


class AvoidFragilesEnv(gym.Env, PybulletInterface):
    metadata = {'render.modes': ['human']}

    def __init__(self, render = False):
        gym.Env.__init__(self)

        self.lidar_points = 32
        PybulletInterface.__init__(self, render = render, lidar_points = self.lidar_points)

        self.action_space       = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=numpy.float32)
        self.observation_space  = spaces.Box(low=-1.0, high=1.0, shape=(6, self.lidar_points), dtype=numpy.float32)

        
    def step(self, action):

        self.step_interface()
        
        vl = 50.0*numpy.clip(action[0], -1.0, 1.0)
        vr = 50.0*numpy.clip(action[1], -1.0, 1.0)

        self.robots[0].set_velocity(vl, vr)
        
        reward  = 0.0
        done    = False

        if self.on_target(0, 0):
            reward = 1.0
            done   = True 
        elif self.on_fragile(0):
            reward = -0.1
        elif self.on_hazard(0):
            reward = -1.0
            done   = True
        elif self.out_board(0):
            reward = -1.0
            done   = True
        elif self.steps > 2000:
            reward = -1.0
            done   = True

        self.pb_client.stepSimulation()

        return self._update_observation(robot_id=0, lidar_points=self.lidar_points), reward, done, None

        
    
    def reset(self):
        robots_count    = 1 
        targets_count   = 1
        hazards_count   = 4
        obstacles_count = 1
        fragile_count   = 8
        moving_count    = 0
        foods_count     = 0
        
        self.reset_interface(targets_count, robots_count, hazards_count, obstacles_count, fragile_count,  moving_count, foods_count)

        return self._update_observation(robot_id=0, lidar_points=self.lidar_points)
        
    def render(self):
        pass

    def close(self):
        pass

    def _update_observation(self, robot_id, lidar_points):
        lidar   = self.get_lidar(robot_id)

        vl, vr  = self.robots[robot_id].get_wheel_velocity()

        result    = numpy.zeros((6, lidar_points), dtype=numpy.float32)

        result[0] = numpy.tanh(vl*numpy.ones(lidar_points)/50.0) #robot velocity, squeezed by tanh
        result[1] = numpy.tanh(vr*numpy.ones(lidar_points)/50.0)
        result[2] = lidar[3]        #obstacles lidar
        result[3] = lidar[1]        #target lidar
        result[4] = lidar[2]        #hazards lidar
        result[5] = lidar[4]        #fragiles lidar
 
        return result

    def _dummy_follow(self):
        items_r, items_yaw = self.get_items_relative_position(self.robots[0].pb_robot, self.targets)


        if numpy.abs(items_yaw[0]) > 0.3:
            if items_yaw[0] > 0.0:
                self.robots[0].set_velocity(-5.0, 5.0)
            else:
                self.robots[0].set_velocity(5.0, -5.0)
        else:
            self.robots[0].set_velocity(50.0, 50.0)
 

    '''
    def render_lidar(self, lidar, size = 256):
        image = Image.new('RGB', (size, size))

        radius  = (256//2) - 10
        center  = size//2
        draw    = ImageDraw.Draw(image)

        self._draw_circle(draw, 0 + center, 0 + center, radius, color=(10, 10, 10))

        count = len(lidar)

        for i in range(count):
            phi = 2.0*numpy.pi*i*1.0/count + 1.5*numpy.pi
            
            if lidar[i] > 0.0:
                distance = lidar[i]*radius

                x = center + distance*numpy.cos(phi)
                y = center + distance*numpy.sin(phi)

                self._draw_circle(draw, int(x), int(y), radius*1.0/count + 2, color=(100, 10, 10))

        
        rgb = cv2.cvtColor(numpy.array(image),cv2.COLOR_BGR2RGB)



        cv2.imshow("cv window", rgb)  
        cv2.waitKey(1)
    
    def _draw_circle(self, draw, x, y, r, color):
        draw.ellipse((x - r, y - r, x + r, y + r), fill = color, outline =color)
    '''
