import gym
from gym import error, spaces, utils
from gym.utils import seeding

from .PybulletClient import *
from .PybulletInterface import *

import numpy
import os

#import cv2

class GoalEnv(gym.Env, PybulletInterface):
    metadata = {'render.modes': ['human']}

    def __init__(self, render = True):
        gym.Env.__init__(self)
        PybulletInterface.__init__(self)

        self.render     = render

        
    def step(self, action):

        self.step_interface()
        

        #self._render_lidar(self.lidar[4])


        items_r, items_yaw = self.get_items_relative_position(self.robots[0].pb_robot, self.targets)


        if numpy.abs(items_yaw[0]) > 0.3:
            if items_yaw[0] > 0.0:
                self.robots[0].set_velocity(-5.0, 5.0)
            else:
                self.robots[0].set_velocity(5.0, -5.0)
        else:
            self.robots[0].set_velocity(50.0, 50.0)
 
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

        if reward != 0:
            print(self.steps, reward)

        self.pb_client.stepSimulation()

        return 0, reward, done, None

        
    
    def reset(self):
        robots_count    = 1 
        targets_count   = 1
        hazards_count   = 3
        obstacles_count = 1
        fragile_count   = 2
        moving_count    = 0
        buttons_count   = 2
        
        self.reset_interface(targets_count, robots_count, hazards_count, obstacles_count, fragile_count,  moving_count, buttons_count)


        
    def render(self):
        pass

    def close(self):
        print('close')

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
