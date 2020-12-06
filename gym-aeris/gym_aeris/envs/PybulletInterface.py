import gym
from gym import error, spaces, utils
from gym.utils import seeding

from .PybulletClient import *
from .RobotBasic import *

import numpy
import os
import cv2

class PybulletInterface():
    def __init__(self, size_type = "small", render = True, lidar_points = 64):
        self.size_type      = size_type
        self.render         = render
        self.lidar_points   = lidar_points
        
        self.pb_client  = PybulletClient(self.render)
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.path_data = self.path + "/../data/"

        
        #self.texture_background = self.pb_client.loadTexture(self.path_data + "background2.png")
        #self.texture_plane      = self.pb_client.loadTexture(self.path_data + "plane.png")
    
    
    def reset_interface(self, targets_count = 1, robots_count = 1, hazards_count = 0, obstacles_count = 0, fragiles_count = 0, movings_count = 0, buttons_count = 0):
        self.path       = os.path.dirname(os.path.abspath(__file__))
        self.pb_client.resetSimulation()

        self.pb_client.setTimeStep(1.0/100.0)
        self.pb_client.setGravity(0, 0, -9.81)

        self.size           = 0.8
        self.plane          = self.pb_client.loadURDF(self.path_data + "base.urdf")

        self.robots     = []
        self.targets    = []
        self.hazards    = []
        self.obstacles  = []
        self.fragiles   = []
        self.movings    = []
        self.buttons    = []

        self.robots_controll = []

        self.positions = self.random_positions(targets_count + robots_count + hazards_count + obstacles_count + fragiles_count + movings_count + buttons_count)


        idx = 0

        for i in range(robots_count):
            x   = self.positions[idx][0]
            y   = self.positions[idx][1]
            z   = 0.01
            yaw = numpy.random.rand()*2.0*numpy.pi

            robot = RobotBasic(self.pb_client, self.path_data + "follower_bot.urdf", x, y, z, yaw)
            self.robots.append(robot)

            idx+= 1
          

        for i in range(targets_count):
            x = self.positions[idx][0]
            y = self.positions[idx][1]
            yaw = numpy.random.rand()*2.0*numpy.pi
            orientation = self.pb_client.getQuaternionFromEuler([0, 0, yaw])
            
            self.targets.append(self.pb_client.loadURDF(self.path_data + "target.urdf", [x, y, 0.06], orientation))
            
            idx+= 1

     
        for i in range(hazards_count):
            x = self.positions[idx][0]
            y = self.positions[idx][1]
            yaw = numpy.random.rand()*2.0*numpy.pi
            orientation = self.pb_client.getQuaternionFromEuler([0, 0, yaw])

            self.hazards.append(self.pb_client.loadURDF(self.path_data + "hazard.urdf", [x, y, 0.06], orientation))

            idx+= 1

        for i in range(obstacles_count):
            x = self.positions[idx][0]
            y = self.positions[idx][1]
            yaw = numpy.random.rand()*2.0*numpy.pi
            orientation = self.pb_client.getQuaternionFromEuler([0, 0, yaw])
            
            self.obstacles.append(self.pb_client.loadURDF(self.path_data + "obstacle.urdf", [x, y, 0.0], orientation))

            idx+= 1

        for i in range(fragiles_count):
            x = self.positions[idx][0]
            y = self.positions[idx][1]
            yaw = numpy.random.rand()*2.0*numpy.pi
            orientation = self.pb_client.getQuaternionFromEuler([0, 0, yaw])
            
            self.fragiles.append(self.pb_client.loadURDF(self.path_data + "fragile.urdf", [x, y, 0.06], orientation))

            idx+= 1

        for i in range(movings_count):
            x = self.positions[idx][0]
            y = self.positions[idx][1]
            yaw = numpy.random.rand()*2.0*numpy.pi
            orientation = self.pb_client.getQuaternionFromEuler([0, 0, yaw])
            
            self.movings.append(self.pb_client.loadURDF(self.path_data + "moving.urdf", [x, y, 0.06], orientation))

            idx+= 1
    
        for i in range(buttons_count):
            x = self.positions[idx][0]
            y = self.positions[idx][1]
            yaw = numpy.random.rand()*2.0*numpy.pi
            orientation = self.pb_client.getQuaternionFromEuler([0, 0, yaw])
            
            self.buttons.append(self.pb_client.loadURDF(self.path_data + "button.urdf", [x, y, 0.0], orientation))

            idx+= 1

        for i in range(100):
            self.pb_client.stepSimulation()
        
        self.steps = 0


    def step_interface(self):
        self._step_movings(self.movings)
        self.lidar = self.get_lidar(0)

        self.steps+=1


    
    '''
    def render_interface(self, mode='human'):
        image = self.pb_client.get_image(0*180.0/numpy.pi - 90, -90.0, 0.0, 0.25, 0, 0, 0, 640, 480)

        rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        cv2.imshow("visualisation", rgb)  
        cv2.waitKey(1)
    '''


    def on_target(self, robot_id, target_id):
        robot_position, robot_angle   = self.robots[robot_id].get_position_and_orientation()
        target_position, target_angle = self.pb_client.getBasePositionAndOrientation(self.targets[target_id])

        dif      = numpy.array(target_position) - numpy.array(robot_position)
        distance = (numpy.sum(dif**2))**0.5

        if distance < 0.08:
            return True
        return False


    def on_fragile(self, robot_id):
        robot_position, robot_angle   = self.robots[robot_id].get_position_and_orientation()

        for i in range(len(self.fragiles)):
            target_position, target_angle = self.pb_client.getBasePositionAndOrientation(self.fragiles[i])

            dif      = numpy.array(target_position) - numpy.array(robot_position)
            distance = (numpy.sum(dif**2))**0.5

            if distance < 0.18:
                return True

        return False


    def on_hazard(self, robot_id):
        robot_position, robot_angle   = self.robots[robot_id].get_position_and_orientation()

        for i in range(len(self.hazards)):
            target_position, target_angle = self.pb_client.getBasePositionAndOrientation(self.hazards[i])

            dif      = numpy.array(target_position) - numpy.array(robot_position)
            distance = (numpy.sum(dif**2))**0.5

            if distance < 0.15:
                return True

        return False

    def out_board(self, robot_id):
        robot_position, _   = self.robots[robot_id].get_position_and_orientation()

        if numpy.abs(robot_position[0]) > 1.0:
            return True

        if numpy.abs(robot_position[1]) > 1.0:
            return True

        return False


    def random_positions(self, count):
        positions = []

        for i in range(count):
            positions.append(self._random_position(positions))

        return numpy.array(positions)
        
    
    def _random_position(self, restricted_positions):
        y = self._rnd()
        x = self._rnd()

        while self._is_restriced(y, x, restricted_positions) == False:
            y = self._rnd()
            x = self._rnd() 

        return [x, y]

    def _is_restriced(self, y, x, restricted_positions, eps = 0.3):     
        for i in range(len(restricted_positions)):
            d = (x - restricted_positions[i][0])**2
            d+= (y - restricted_positions[i][1])**2
            d = d**0.5

            if d < eps:
                return False

        if numpy.abs(x) > 0.85:
            return False

        if numpy.abs(y) > 0.85:
            return False

        return True

    def _rnd(self): 
        return (numpy.random.rand()*2 - 1.0)


    def get_lidar(self, robot_id):
        result = numpy.zeros((7, self.lidar_points))
        
        '''
        if len(self.robots) > 0:
            result[0]     = self._lidar_process(self.robots[robot_id].pb_robot, self.robots)
        '''

        if len(self.targets) > 0:
            result[1]    = self._lidar_process(self.robots[robot_id].pb_robot, self.targets)

        if len(self.hazards) > 0:
            result[2]    = self._lidar_process(self.robots[robot_id].pb_robot, self.hazards)
 
        if len(self.obstacles) > 0:
            result[3]  = self._lidar_process(self.robots[robot_id].pb_robot, self.obstacles)

        if len(self.fragiles) > 0:
            result[4]   = self._lidar_process(self.robots[robot_id].pb_robot, self.fragiles)

        if len(self.movings) > 0:
            result[5]    = self._lidar_process(self.robots[robot_id].pb_robot, self.movings)

        if len(self.buttons) > 0:
            result[6]    = self._lidar_process(self.robots[robot_id].pb_robot, self.buttons)

        return result

    def get_items_relative_position(self, robot, items):
        robot_position, orientation = self.pb_client.getBasePositionAndOrientation(robot)
        robot_angle = self.pb_client.getEulerFromQuaternion(orientation)

        robot_yaw = robot_angle[2]

        items_count     = len(items)
        items_position  = numpy.zeros((items_count, 3))
        for i in range(items_count):
            item_position, _ = self.pb_client.getBasePositionAndOrientation(items[i])
            
            items_position[i][0] = item_position[0] - robot_position[0]
            items_position[i][1] = item_position[1] - robot_position[1]
            items_position[i][2] = item_position[2] - robot_position[2]
            

        items_r, items_yaw = self._polar_position(items_position)

        return items_r, items_yaw - robot_yaw


    def _lidar_process(self, robot, items):
        
        items_r, items_yaw = self.get_items_relative_position(robot, items)
     
        distance = numpy.tanh(items_r)
        idx      = numpy.floor(self.lidar_points*items_yaw/(2.0*numpy.pi)).astype(int)%self.lidar_points

        result   = 10*numpy.ones(self.lidar_points)
        for i in range(len(items)):
            result[idx[i]] = min(distance[i], result[idx[i]])

        result[result > 9] = 0.0

        return result


    def _polar_position(self, position):
        position = numpy.array(position)
        
        r   = numpy.sum(position**2.0, axis=1)**0.5

        position = position.transpose()
        phi = numpy.arctan2(position[1], position[0])

        return r, phi

    def _step_movings(self, items):

        for i in range(len(items)):
            #self.pb_client.applyExternalTorque(items[i], -1, [0, 0, 1], self.pb_client.LINK_FRAME)

            #gemPos, gemOrn = p.getBasePositionAndOrientation(gemId)
            position, _ = self.pb_client.getBasePositionAndOrientation(items[i])
            velocity    = self.pb_client.getBaseVelocity(items[i])

            phi = 2.0*numpy.pi*(self.steps%64)/64.0
            v       = 0.1
            alpha   = 1.0

            vx = 0.01
            vy = 0.01

            fx = alpha*(velocity[0][0] - vx)
            fy = alpha*(velocity[0][1] - vy)


            self.pb_client.applyExternalForce(objectUniqueId=items[i], linkIndex=-1, forceObj=[fx, fy, 0], posObj=position, flags=self.pb_client.WORLD_FRAME)



