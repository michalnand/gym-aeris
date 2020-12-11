import gym
from gym import error, spaces, utils
from gym.utils import seeding

from .PybulletClient import *
from .RobotBasic import *

import numpy
import os


import cv2
from PIL import Image, ImageDraw

class PybulletInterface():
    def __init__(self, render = True, lidar_points = 32):
        self.render         = render
        self.lidar_points   = lidar_points
        
        self.pb_client  = PybulletClient(self.render)
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.path_data = self.path + "/../data/"

        
        #self.texture_background = self.pb_client.loadTexture(self.path_data + "background2.png")
        #self.texture_plane      = self.pb_client.loadTexture(self.path_data + "plane.png")
    
    
    def reset_interface(self, targets_count = 1, robots_count = 1, hazards_count = 0, obstacles_count = 0, fragiles_count = 0, movings_count = 0, foods_count = 0):
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
        self.foods      = []

        self.robots_controll = []

        self.positions = self.random_positions(targets_count + robots_count + hazards_count + obstacles_count + fragiles_count + movings_count + foods_count)


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
            
            self.obstacles.append(self.pb_client.loadURDF(self.path_data + "obstacle.urdf", [x, y, 0.501], orientation))

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
    
        for i in range(foods_count):
            x = self.positions[idx][0]
            y = self.positions[idx][1]
            yaw = numpy.random.rand()*2.0*numpy.pi
            orientation = self.pb_client.getQuaternionFromEuler([0, 0, yaw])
            
            self.foods.append(self.pb_client.loadURDF(self.path_data + "food.urdf", [x, y, 0.06], orientation))

            idx+= 1

        for i in range(100):
            self.pb_client.stepSimulation()
        
        self.steps = 0


    def step_interface(self):

        #self._step_movings(self.movings)
        self.lidar = self.get_lidar(0)

        self.steps+=1

        #self.render_lidar(self.lidar)


    
    '''
    def render_interface(self, mode='human'):
        image = self.pb_client.get_image(0*180.0/numpy.pi - 90, -90.0, 0.0, 0.25, 0, 0, 0, 640, 480)

        rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        cv2.imshow("visualisation", rgb)  
        cv2.waitKey(1)
    '''


    def on_target(self, robot_id, target_id):
        distance = self.target_distance(robot_id, target_id)
        
        if distance < 0.15:
            return True
        return False

    def target_distance(self, robot_id = 0, target_id = 0):
        robot_position, robot_angle   = self.robots[robot_id].get_position_and_orientation()
        target_position, target_angle = self.pb_client.getBasePositionAndOrientation(self.targets[target_id])

        dif      = numpy.array(target_position) - numpy.array(robot_position)
        distance = (numpy.sum(dif**2))**0.5

        return distance


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

    def on_food(self, robot_id):
        robot_position, robot_angle   = self.robots[robot_id].get_position_and_orientation()

        for i in range(len(self.foods)):
            target_position, target_angle = self.pb_client.getBasePositionAndOrientation(self.foods[i])

            dif      = numpy.array(target_position) - numpy.array(robot_position)
            distance = (numpy.sum(dif**2))**0.5

            if distance < 0.15:
                return i

        return -1


    def closest_food_distance(self, robot_id = 0):
        min_distance = 1000
        robot_position, robot_angle   = self.robots[robot_id].get_position_and_orientation()

        for i in range(len(self.foods)):
            target_position, target_angle = self.pb_client.getBasePositionAndOrientation(self.foods[i])

            dif      = numpy.array(target_position) - numpy.array(robot_position)
            distance = (numpy.sum(dif**2))**0.5

            if distance < min_distance:
                min_distance = distance


        return min_distance

  

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

        if len(self.foods) > 0:
            result[6]    = self._lidar_process(self.robots[robot_id].pb_robot, self.foods)


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

    def get_items_bounding_box(self, items):
        result = numpy.zeros((len(items), 2, 3))
        
        for i in range(len(items)):
            v = self.pb_client.getAABB(items[i])
            
            result[i][0] = numpy.array(v[0])
            result[i][1] = numpy.array(v[1])
        
        return result

    def _lidar_process(self, robot, items):

        robot_position, orientation = self.pb_client.getBasePositionAndOrientation(robot)
        robot_angle = self.pb_client.getEulerFromQuaternion(orientation)

        bb               = self.get_items_bounding_box(items)
        
        robot_position = numpy.array(robot_position)
        items_r, items_yaw = self._get_scan_lines(robot_position, bb)

        '''
        if self.steps%150 == 0:
            self._draw_scan_lines(robot_position, items_r, items_yaw)
        '''
        
        items_yaw = items_yaw - robot_angle[2]
        items_yaw = self._map_angle(items_yaw)

        distance = numpy.tanh(items_r)
        idx      = (numpy.round(self.lidar_points*items_yaw/(2.0*numpy.pi)).astype(int) + self.lidar_points//2)%self.lidar_points

        result   = numpy.ones(self.lidar_points)
        for i in range(len(items_r)):
            result[idx[i]] = min(distance[i], result[idx[i]])


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

            phi     = 2.0*numpy.pi*(self.steps%64)/64.0
            v       = 0.1
            alpha   = 1.0

            vx = 0.01
            vy = 0.01

            fx = alpha*(velocity[0][0] - vx)
            fy = alpha*(velocity[0][1] - vy)

            self.pb_client.applyExternalForce(objectUniqueId=items[i], linkIndex=-1, forceObj=[fx, fy, 0], posObj=position, flags=self.pb_client.WORLD_FRAME)


    def _get_scan_lines(self, center_position, bounding_boxes, interpolation_steps = 10):
        items_count         = bounding_boxes.shape[0]
        center              = center_position.reshape(1, 1, 3).repeat(2, axis=1).repeat(items_count, axis=0)
        relative_position   = bounding_boxes - center_position
        
        #remove z-axis
        relative_position   = numpy.delete(relative_position, 2, axis=2)

        #vector length
        r = ((relative_position**2).sum(2))**0.5

        #angle
        relative_position   = relative_position.reshape(items_count*2, 2).transpose()
        yaw = numpy.arctan2(relative_position[1], relative_position[0])

        yaw   = yaw.reshape(items_count, 2)

        r   = r.transpose()
        yaw = yaw.transpose()

        r_interpolated   = numpy.zeros((interpolation_steps, r.shape[1]))
        yaw_interpolated = numpy.zeros((interpolation_steps, yaw.shape[1]))

      
        r_start = r[0]
        r_end   = r[1]
        for i in range(interpolation_steps):
            w = i/interpolation_steps
            r_interpolated[i] = (1.0 - w)*r_start + w*r_end

        yaw_start = yaw[0]
        yaw_end   = yaw[1]
        for i in range(interpolation_steps):
            w = i/interpolation_steps
            yaw_interpolated[i] = (1.0 - w)*yaw_start + w*yaw_end

        r_interpolated      = r_interpolated.transpose().reshape(interpolation_steps*items_count)
        yaw_interpolated    = yaw_interpolated.transpose().reshape(interpolation_steps*items_count)

        return r_interpolated, self._map_angle(yaw_interpolated)

    def _draw_scan_lines(self, origin, r, yaw, color = [1, 0, 0]):
        
        origin[2] = 0.02
        count = r.shape[0]
        for i in range(count):
            r_      = r[i]
            yaw_    = yaw[i]

            f = [origin[0], origin[1], origin[2]]
            t = [origin[0] + r_*numpy.cos(yaw_), origin[1] + r_*numpy.sin(yaw_), origin[2]]
            self.pb_client.addUserDebugLine(f,t, color)


    def render_lidar(self, lidar, size = 256):
        image = Image.new('RGB', (size, size))

        radius  = (256//2) - 10
        center  = size//2
        draw    = ImageDraw.Draw(image)

        self._draw_circle(draw, 0 + center, 0 + center, radius, color=(10, 10, 10))

        items_types = lidar.shape[0]
        

        for j in range(items_types):
            phi = 2.0*numpy.pi*j/items_types
            r = int(128*(1 + numpy.sin(phi + 0.0*numpy.pi/3)))
            g = int(128*(1 + numpy.sin(phi + 1.0*numpy.pi/3)))
            b = int(128*(1 + numpy.sin(phi + 2.0*numpy.pi/3)))

            for i in range(lidar.shape[1]):
                count = lidar.shape[1]
                
                if lidar[j][i] > 0.0:
                    phi = 2.0*numpy.pi*i*1.0/count - numpy.pi*0.5 + numpy.pi

                    distance = lidar[j][i]*radius

                    x = center + distance*numpy.cos(phi)
                    y = center + distance*numpy.sin(phi)

                    self._draw_circle(draw, int(x), int(y), radius*1.0/count + 2, color=(r, g, b))

        
        rgb = cv2.cvtColor(numpy.array(image),cv2.COLOR_BGR2RGB)

        cv2.imshow("cv window", rgb)  
        cv2.waitKey(1)
    
    def _draw_circle(self, draw, x, y, r, color):
        draw.ellipse((x - r, y - r, x + r, y + r), fill = color, outline =color)

    def _map_angle(self, angle):
        return (angle + 2 * numpy.pi) % (2 * numpy.pi)