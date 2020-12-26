import gym
from gym import error, spaces, utils
from gym.utils import seeding

from .PybulletClient    import *
from .RobotBasic        import *

import numpy
import os


class SwarmFoodGatheringEnv(gym.Env):

    def __init__(self, render = False):
        gym.Env.__init__(self) 

        self.robots_count           = 32
        self.foods_count            = 8*self.robots_count
        self.fragiles_count         = 2*self.robots_count

        self.closest_count          = 3
        self.internal_state_size    = 4

        view_camera_distance        = 1.05
        view_camera_angle           = -89.9

        self.pb_client  = PybulletClient(render, view_camera_distance, view_camera_angle)
        self.path       = os.path.dirname(os.path.abspath(__file__))
        self.path_data  = self.path + "/../data/"

        self.actions_shape      = (self.robots_count*(2 + self.internal_state_size), )
        self.observation_shape  = (self.robots_count, self.closest_count*(2 + 2 + 2 + self.internal_state_size))
        
        self.action_space       = spaces.Box(low=-1.0, high=1.0, shape=self.actions_shape, dtype=numpy.float32)
        self.observation_space  = spaces.Box(low=-1.0, high=1.0, shape=self.observation_shape, dtype=numpy.float32)

    def reset(self):
        self.steps = 0
        self.pb_client.resetSimulation()

        self.pb_client.setTimeStep(1.0/100.0)
        self.pb_client.setGravity(0, 0, -9.81)

        self.board_size     = 1.0
        self.plane          = self.pb_client.loadURDF(self.path_data + "base_2.urdf")

        self.robots     = []
        self.foods      = []
        self.fragiles   = []

        positions = self.random_positions(self.foods_count, self.robots_count + self.fragiles_count)

        idx = 0

        for i in range(self.foods_count):
            x = positions[idx][0]
            y = positions[idx][1]
            yaw = numpy.random.rand()*2.0*numpy.pi
            orientation = self.pb_client.getQuaternionFromEuler([0, 0, yaw])
            
            self.foods.append(self.pb_client.loadURDF(self.path_data + "food_small.urdf", [x, y, 0.02], orientation))

            idx+= 1

        for i in range(self.robots_count):
            x = positions[idx][0]
            y = positions[idx][1]
            yaw = numpy.random.rand()*2.0*numpy.pi

            robot = RobotBasic(self.pb_client, self.path_data + "robot_blue.urdf", x, y, 0.02, yaw, 1, 2, scale=0.3)
            self.robots.append(robot)
        
            idx+= 1

        for i in range(self.fragiles_count):
            x = positions[idx][0]
            y = positions[idx][1]
            yaw = numpy.random.rand()*2.0*numpy.pi
            orientation = self.pb_client.getQuaternionFromEuler([0, 0, yaw])
            
            self.fragiles.append(self.pb_client.loadURDF(self.path_data + "fragile_small.urdf", [x, y, 0.02], orientation))

            idx+= 1

        for i in range(100):
            self.pb_client.stepSimulation()

        self._update_distances()

        self.internal_state = numpy.zeros((self.robots_count, self.internal_state_size))

        return self._update_observation()
        

    def step(self, action):
        self.steps+= 1

        action_ = action.reshape((self.robots_count, 2 + self.internal_state_size))

        for i in range(self.robots_count):
            vl = 50.0*numpy.clip(action_[i][0], -1.0, 1.0)
            vr = 50.0*numpy.clip(action_[i][1], -1.0, 1.0)

            self.robots[i].set_velocity(vl, vr)
        
        '''
        for i in range(self.robots_count):
            self._dummy_follow(i)
        '''

        self.internal_state = action_[:, 2:2+self.internal_state_size].copy()
        
        for i in range(4):
            self.pb_client.stepSimulation()

        self._update_distances()
     
        reward = -0.01
        done   = False

        collected, food_present = self._eat_food()

        fragiles_contacts = self._fragile_colission()

        reward+= -0.1*fragiles_contacts
 
        if self.steps >= 1000:
            done    = True
            reward  = -1.0
        elif self._out_board():
            done    = True
            reward  = -1.0
        elif collected > 0:
            reward+= collected
        
        if food_present == False:
            reward+= 1
            done = True
            

        return self._update_observation(), reward, done, None

        
    def render(self):
        pass

    def close(self):
        pass

    def _update_distances(self):
        self.robots_positions, self.robots_angles   = self._get_robots_positions()
        self.foods_positions                        = self._get_foods_positions()
        self.fragiles_positions                     = self._get_fragiles_positions()

        self.robots_robots_distances    = numpy.abs(self.robots_positions.transpose() - self.robots_positions)
        self.robots_foods_distances     = numpy.abs(self.robots_positions.transpose() - self.foods_positions)
        self.robots_fragiles_distances  = numpy.abs(self.robots_positions.transpose() - self.fragiles_positions)

    def _update_observation(self):
        
        #find indices of closest N robots, but skip robot itself (zero distance)
        closest_robots = self.robots_robots_distances.argsort()[:, 1:self.closest_count+1]

        #find indices of closest N foods
        closest_foods = self.robots_foods_distances.argsort()[:, :self.closest_count]
        
        if closest_foods.shape[1] != self.closest_count:
            dif   = self.closest_count - closest_foods.shape[1] 
            align = numpy.zeros((self.robots_count, dif), dtype=int)
            closest_foods = numpy.hstack([closest_foods, align])


        #find indices of closest N fragiles
        closest_fragiles = self.robots_fragiles_distances.argsort()[:, :self.closest_count]

        #take the closest positions
        tmp = numpy.tile(self.robots_positions, (self.robots_count, 1) )
        closest_robots_positions = numpy.take(tmp, closest_robots)
        
        tmp = numpy.tile(self.foods_positions, (self.robots_count, 1) )
        closest_foods_positions = numpy.take(tmp, closest_foods)

        tmp = numpy.tile(self.fragiles_positions, (self.robots_count, 1) )
        closest_fragiles_positions = numpy.take(tmp, closest_fragiles)
 
        #convert cartesian positions into polar coordinates, relative to robot positiom

        relative_robots_positions   = closest_robots_positions - self.robots_positions.transpose()
        robots_r                    = numpy.abs(relative_robots_positions)
        robots_angles               = numpy.angle(relative_robots_positions)

        robots_relative_angles      = robots_angles - numpy.expand_dims(self.robots_angles, 1)
        robots_relative_angles      = self._map_angle(robots_relative_angles)


        relative_foods_positions    = closest_foods_positions - self.robots_positions.transpose()
        foods_r                     = numpy.abs(relative_foods_positions)
        foods_angles                = numpy.angle(relative_foods_positions)

        foods_relative_angles       = foods_angles - numpy.expand_dims(self.robots_angles, 1)
        foods_relative_angles       = self._map_angle(foods_relative_angles)


        relative_fragiles_positions    = closest_fragiles_positions - self.robots_positions.transpose()
        fragiles_r                     = numpy.abs(relative_fragiles_positions)
        fragiles_angles                = numpy.angle(relative_fragiles_positions)

        fragiles_relative_angles       = fragiles_angles - numpy.expand_dims(self.robots_angles, 1)
        fragiles_relative_angles       = self._map_angle(fragiles_relative_angles)


        #take internal state, using closest robot indices
        #internal_state_ = (robot, other_robots_features) = (robot, self.closest_count*self.internal_state_size)
        internal_state_     = self.internal_state.take(closest_robots, axis=0)        
        internal_state_     = internal_state_.reshape((self.robots_count, self.closest_count*self.internal_state_size))

        

        '''
        put together into state vector
        state = (robot, values)
        values : robots_r[robot_id], robots_relative_angles[robot_id], foods_r[robot_id], foods_relative_angles[robot_id], internal_state[robot_id]
        '''

        robots_r_               = numpy.tanh(robots_r)
        robots_relative_angles_ = numpy.tanh( (robots_relative_angles - numpy.pi)/numpy.pi )
        #robots_relative_angles_ = robots_relative_angles

        foods_r_               = numpy.tanh(foods_r)
        foods_relative_angles_ = numpy.tanh( (foods_relative_angles - numpy.pi)/numpy.pi )
        #foods_relative_angles_  = foods_relative_angles

        fragiles_r_               = numpy.tanh(fragiles_r)
        fragiles_relative_angles_ = numpy.tanh( (fragiles_relative_angles - numpy.pi)/numpy.pi )

        self.observation    = numpy.hstack((robots_r_, robots_relative_angles_, foods_r_, foods_relative_angles_, fragiles_r_, fragiles_relative_angles_, internal_state_))

        '''
        if self.steps%10 == 0:
            for j in range(robots_r.shape[0]):
                for i in range(robots_r.shape[1]):
            
                    source = self.robots_positions[0][j]
                    sx     = source.real
                    sy     = source.imag

                    angle  = robots_relative_angles[j][i] + self.robots_angles[j]

                    dx     = sx + robots_r[j][i]*numpy.cos(angle)
                    dy     = sy + robots_r[j][i]*numpy.sin(angle)

                    self.pb_client.addUserDebugLine([sx, sy, 0.01], [dx, dy, 0.01], [0, 0, 1])

            for j in range(foods_r.shape[0]):
                for i in range(foods_r.shape[1]):
            
                    source = self.robots_positions[0][j]
                    sx     = source.real
                    sy     = source.imag

                    angle  = foods_relative_angles[j][i] + self.robots_angles[j]

                    dx     = sx + foods_r[j][i]*numpy.cos(angle)
                    dy     = sy + foods_r[j][i]*numpy.sin(angle)

                    self.pb_client.addUserDebugLine([sx, sy, 0.01], [dx, dy, 0.01], [1, 0, 0])

            for j in range(fragiles_r.shape[0]):
                for i in range(fragiles_r.shape[1]):
            
                    source = self.robots_positions[0][j]
                    sx     = source.real
                    sy     = source.imag

                    angle  = fragiles_relative_angles[j][i] + self.robots_angles[j]

                    dx     = sx + fragiles_r[j][i]*numpy.cos(angle)
                    dy     = sy + fragiles_r[j][i]*numpy.sin(angle)

                    self.pb_client.addUserDebugLine([sx, sy, 0.01], [dx, dy, 0.01], [0, 1, 0])
        '''

        #print(self.observation)

        return self.observation


    def random_positions(self, foods, other):
        positions = []

        cy      = 0.9*self._rnd()*self._max_size()
        cx      = 0.9*self._rnd()*self._max_size()
        sigma   = 0.4 

        for i in range(foods):
            positions.append(self._random_position_centered(positions, cy, cx, sigma, 0.026))

        for i in range(other):
            positions.append(self._random_position(positions))

        return numpy.array(positions)

    def _random_position_centered(self, restricted_positions, cy, cx, sigma, distance_min):
        y = cy + sigma*self._rndn()*self._max_size()
        x = cx + sigma*self._rndn()*self._max_size()
 
        while self._is_restriced(y, x, restricted_positions, distance_min) == False:
            y = cy + sigma*self._rndn()*self._max_size()
            x = cx + sigma*self._rndn()*self._max_size()

        return [x, y]

    def _random_position(self, restricted_positions):
        y = self._rnd()*self._max_size()
        x = self._rnd()*self._max_size()
 
        while self._is_restriced(y, x, restricted_positions) == False:
            y = self._rnd()*self._max_size()
            x = self._rnd()*self._max_size()

        return [x, y]

    def _is_restriced(self, y, x, restricted_positions, eps = 0.04):     
        for i in range(len(restricted_positions)):
            d = (x - restricted_positions[i][0])**2
            d+= (y - restricted_positions[i][1])**2
            d = d**0.5

            if d < eps:
                return False

        if numpy.abs(x) > self._max_size():
            return False

        if numpy.abs(y) > self._max_size():
            return False

        return True

    def _rnd(self): 
        return (numpy.random.rand()*2 - 1.0)

    def _rndn(self): 
        return numpy.random.randn()

    def _max_size(self):
        return self.board_size*0.95

    def _out_board(self):
        for robot_id in range(self.robots_count):
            robot_position, _   = self.robots[robot_id].get_position_and_orientation()

            if numpy.abs(robot_position[0]) > self.board_size:
                return True

            if numpy.abs(robot_position[1]) > self.board_size:
                return True

        return False

    def _get_robots_positions(self):
        positions = numpy.zeros((1, self.robots_count), dtype=numpy.complex)
        angles    = numpy.zeros(self.robots_count)

        for i in range(self.robots_count):
            x, y, z, pitch, roll, yaw   = self.robots[i].get_position()
            positions[0][i]             = x + y*1.0j
            angles[i]                   = yaw

        return positions, angles
    
    def _get_foods_positions(self):
        foods_count = len(self.foods) 
        positions = numpy.zeros((1, foods_count), dtype=numpy.complex)

        for i in range(foods_count):
            position, _ = self.pb_client.getBasePositionAndOrientation(self.foods[i])
            
            positions[0][i] = position[0] + position[1]*1.0j

        return positions


    def _get_fragiles_positions(self):
        fragiles_count = len(self.fragiles) 
        positions = numpy.zeros((1, fragiles_count), dtype=numpy.complex)

        for i in range(fragiles_count):
            position, _ = self.pb_client.getBasePositionAndOrientation(self.fragiles[i])
            
            positions[0][i] = position[0] + position[1]*1.0j

        return positions

    def _eat_food(self):
        eat_mask = self.robots_foods_distances < 0.04
        eat_mask = eat_mask.any(axis=0)
        
        count   = eat_mask.sum()

        if count > 0:
            remove_indices = numpy.where(eat_mask == True)[0]
            
            for i in range(len(remove_indices)):
                self.pb_client.removeBody(self.foods[remove_indices[i]])
            
            foods = []
            for i in range(len(self.foods)):
                if eat_mask[i] == False:
                    foods.append(self.foods[i])

            self.foods = foods

        if len(self.foods) != 0:
            food_present = True
        else:
            food_present = False

        return count, food_present

    def _fragile_colission(self):
        contact_mask = self.robots_fragiles_distances < 0.038
        
        count = contact_mask.sum()

        return count

    def _map_angle(self, angle):
        return (angle + 2 * numpy.pi) % (2 * numpy.pi)


    def _dummy_follow(self, robot_id):

        #self.observation    = numpy.hstack((robots_r_, robots_relative_angles_, foods_r_, foods_relative_angles_, internal_state_))
        s = self.observation[robot_id]

        yaw = s[9]*numpy.pi

        if numpy.abs(yaw) > 0.2: 
            #if yaw < numpy.pi:
            if yaw < 0.0:
                self.robots[robot_id].set_velocity(-8.0, 8.0)
            else:
                self.robots[robot_id].set_velocity(8.0, -8.0)
        else:
            self.robots[robot_id].set_velocity(50.0, 50.0)
        

        
 

       
       