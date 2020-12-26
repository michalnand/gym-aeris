import gym
from gym import error, spaces, utils
from gym.utils import seeding

from .PybulletClient    import *
from .RobotBasic        import *

import numpy
import os


class SwarmCaptureTheFlagEnv(gym.Env):

    def __init__(self, render = False):
        gym.Env.__init__(self) 

        self.robots_count           = 16
        self.obstacles_count        = 20

        self.closest_count          = 3
        self.internal_state_size    = 4

        view_camera_distance        = 1.05
        view_camera_angle           = -89.9

        self.pb_client  = PybulletClient(render, view_camera_distance, view_camera_angle)
        self.path       = os.path.dirname(os.path.abspath(__file__))
        self.path_data  = self.path + "/../data/"

        self.actions_shape      = (self.robots_count*(2 + self.internal_state_size), )
        self.observation_shape  = (2, self.robots_count, self.closest_count*(2 + 2 + 2 + self.internal_state_size))
        
        self.action_space       = spaces.Box(low=-1.0, high=1.0, shape=self.actions_shape, dtype=numpy.float32)
        self.observation_space  = spaces.Box(low=-1.0, high=1.0, shape=self.observation_shape, dtype=numpy.float32)

    def reset(self):
        self.steps = 0
        self.pb_client.resetSimulation()

        self.pb_client.setTimeStep(1.0/100.0)
        self.pb_client.setGravity(0, 0, -9.81)

        self.board_size     = 1.0
        self.plane          = self.pb_client.loadURDF(self.path_data + "base_2.urdf")

        self.blue_flag      = []
        self.blue_robots    = []

        self.red_flag       = []
        self.red_robots     = []
      


        self.obstacles = []
        obstacles_positions = self._make_obstacles()
        for i in range(self.obstacles_count):

            x = obstacles_positions[i][0]
            y = obstacles_positions[i][1]

            obstacle_a = self.pb_client.loadURDF(self.path_data + "obstacle_cube.urdf", [x, y, 0.025])
            obstacle_b = self.pb_client.loadURDF(self.path_data + "obstacle_cube.urdf", [x, -y, 0.025])
            obstacle_c = self.pb_client.loadURDF(self.path_data + "obstacle_cube.urdf", [-x, y, 0.025])
            obstacle_d = self.pb_client.loadURDF(self.path_data + "obstacle_cube.urdf", [-x, -y, 0.025])

            self.obstacles.append(obstacle_a)
            self.obstacles.append(obstacle_b)
            self.obstacles.append(obstacle_c)
            self.obstacles.append(obstacle_d)
        


        blue_flag_position, blue_robots_positions, blue_robots_angles = self._make_base()

        red_robots_positions    = -1.0*blue_robots_positions
        red_robots_angles       = blue_robots_angles + numpy.pi

        red_flag_position       = -1.0*blue_flag_position

        self.blue_flag.append(self.pb_client.loadURDF(self.path_data + "target_blue.urdf", [blue_flag_position[0], blue_flag_position[1], 0.06]))
        for i in range(self.robots_count):
            x   = blue_robots_positions[i][0]
            y   = blue_robots_positions[i][1]
            yaw = blue_robots_angles[i]

            robot = RobotBasic(self.pb_client, self.path_data + "robot_blue.urdf", x, y, 0.04, yaw, 1, 2, scale=0.3)
            self.blue_robots.append(robot)
 

        self.red_flag.append(self.pb_client.loadURDF(self.path_data + "target_red.urdf", [red_flag_position[0], red_flag_position[1], 0.06]))
        for i in range(self.robots_count):
            x   = red_robots_positions[i][0]
            y   = red_robots_positions[i][1]
            yaw = red_robots_angles[i]

            robot = RobotBasic(self.pb_client, self.path_data + "robot_red.urdf", x, y, 0.04, yaw, 1, 2, scale=0.3)
            self.red_robots.append(robot)


        


        for i in range(100):
            self.pb_client.stepSimulation()



        return self._update_observation()
        

    def step(self, action):
        self.steps+= 1

        '''
        action_ = action.reshape((self.robots_count, 2 + self.internal_state_size))

        for i in range(self.robots_count):
            vl = 50.0*numpy.clip(action_[i][0], -1.0, 1.0)
            vr = 50.0*numpy.clip(action_[i][1], -1.0, 1.0)

            self.robots[i].set_velocity(vl, vr)
        
        self.internal_state = action_[:, 2:2+self.internal_state_size].copy()
        '''

        for i in range(4):
            self.pb_client.stepSimulation()
     
        reward = 0.0
        done   = False

        if self.steps >= 1000:
            done    = True
            reward  = -1.0
        
            

        return self._update_observation(), reward, done, None

        
    def render(self):
        pass

    def close(self):
        pass

    def _update_observation(self):
        
        return numpy.zeros(self.observation_shape)



    
  


    def _make_base(self):
        cx  = self._max_size()
        cy  = self._max_size()
        flag_position       = numpy.zeros(2)
        flag_position[0]    = cx
        flag_position[1]    = cy

        robots_positions = self._random_gaussian_positions(self.robots_count, cx, cy, 0.1)

        robots_angles    = (numpy.random.rand(self.robots_count)*2.0 - 1.0)*numpy.pi

        return flag_position, robots_positions, robots_angles


    def _make_obstacles(self):
        max_x  = self._max_size()*0.98
        max_y  = self._max_size()*0.98
     
        positions = self._random_uniform_positions(self.obstacles_count, max_x, max_y)

        return positions


  

    def _random_gaussian_positions(self, count, cx, cy, sigma):
        positions = [[self._max_size(), self._max_size()]]

        for i in range(count):
            positions.append(self._random_guassian_position(positions, cx, cy, sigma))

        positions = positions[1:]

        return numpy.array(positions)

     
    def _random_guassian_position(self, restricted_positions, cx, cy, sigma):
        x = self._rndn()*sigma + cx
        y = self._rndn()*sigma + cy
        
        while self._is_restriced(x, y, restricted_positions) == False:
            x = self._rndn()*sigma + cx
            y = self._rndn()*sigma + cy
            
        return [x, y]


    def _random_uniform_positions(self, count, max_x, max_y):
        positions = []

        for i in range(count):
            positions.append(self._random_position(positions, max_x, max_y))

        return numpy.array(positions)

     
    def _random_position(self, restricted_positions, max_x, max_y):
        x = self._rnd()*max_y
        y = self._rnd()*max_x
        
        while self._is_restriced(x, y, restricted_positions) == False:
            x = self._rnd()*max_y
            y = self._rnd()*max_x
            
        return [x, y]

    def _is_restriced(self, x, y, restricted_positions, eps = 0.06):     
        for i in range(len(restricted_positions)):
            d = (y - restricted_positions[i][1])**2
            d+= (x - restricted_positions[i][0])**2
            d = d**0.5

            if d < eps:
                return False

        if numpy.abs(x) > self._max_size():
            return False

        if numpy.abs(y) > self._max_size():
            return False

        return True

    def _rndn(self):
        return numpy.random.randn()

    def _rnd(self):
        return numpy.random.rand()

    def _max_size(self):
        return 0.8*self.board_size