from .PybulletClient import *
from .RobotBasic import *
import numpy
import os

class GridInterface:
    def __init__(self, render = False, view_camera_distance = 1.5, view_camera_angle = -80.0):
        self.render     = render
        
        self.pb_client  = PybulletClient(self.render, view_camera_distance, view_camera_angle)
        self.path       = os.path.dirname(os.path.abspath(__file__))
        self.path_data  = self.path + "/../data/"

    
    def reset_interface(self, grid_map):
        self.path       = os.path.dirname(os.path.abspath(__file__))
        self.pb_client.resetSimulation()

        self.pb_client.setTimeStep(1.0/100.0)
        self.pb_client.setGravity(0, 0, -9.81)

        self.board_size     = 1.0
        self.plane          = self.pb_client.loadURDF(self.path_data + "base_2.urdf")

        self.item_max       = numpy.max(grid_map)

        self.make_grid(grid_map)

        for _ in range(100):
            self.pb_client.stepSimulation()
        
        self.steps = 0

    def step_interface(self, action, robot_id = 0, sim_steps = 4):
        vl = 50.0*numpy.clip(action[0], -1.0, 1.0)
        vr = 50.0*numpy.clip(action[1], -1.0, 1.0)

        self.robots[robot_id].set_velocity(vl, vr)

        for _ in range(sim_steps):
            self.pb_client.stepSimulation()

        self.steps+=1
            
    def make_grid(self, grid_map):
        height = len(grid_map)
        width  = len(grid_map[0])
        scale  = 1.7

        self.robots         = []
        self.cubes          = []
        self.targets        = []
        self.targets_blue   = []
        self.targets_red    = []

        self.items = []
    
        for y in range(height):
            for x in range(width):
                item_id = grid_map[y][x]
                if item_id != 0:
                    item = self.put_item(item_id, y, x, height, width, scale)
                    self.items.append(item)

        self.items = numpy.array(self.items)

   
    def put_item(self, item_id, y, x, height, width, scale):
        x_pos = scale*(x/height - 0.5)
        y_pos = scale*(y/width  - 0.5)

        yaw     = 0.0
        active  = 1.0
        z_pos   = 0.0

        if item_id == 1:
            element = RobotBasic(self.pb_client, self.path_data + "robot_blue.urdf", x_pos, y_pos, z_pos, 0.01)
            self.robots.append(element)

        if item_id == 2:
            element = self.pb_client.loadURDF(self.path_data + "obstacle_cube.urdf", [x_pos, y_pos, 0.055])
            self.cubes.append(element)

        if item_id == 3:
            element = self.pb_client.loadURDF(self.path_data + "target.urdf", [x_pos, y_pos, 0.055])
            self.targets.append(element)

        if item_id == 4:
            element = self.pb_client.loadURDF(self.path_data + "target_blue.urdf", [x_pos, y_pos, 0.055])
            self.targets_blue.append(element)

        if item_id == 5:
            element = self.pb_client.loadURDF(self.path_data + "target_red.urdf", [x_pos, y_pos, 0.055])
            self.targets_red.append(element)
        
        item = [x_pos, y_pos, z_pos, yaw, item_id/self.item_max, active]

        return item

    def update_observation(self, robot_id = 0):

        observation = self.items.copy()
        observation = numpy.transpose(observation)

        x, y, z, _, _, _ = self.robots[robot_id].get_position()

        #relative position, to robot
        observation[0] = observation[0] - x
        observation[1] = observation[1] - y
        observation[2] = observation[2] - z

        #observation = numpy.clip(0.5*observation + 1.0, 0.0, 1.0)
        return observation

    def on_target(self, robot_id, target_id):
        distance = self.target_distance(robot_id, target_id)
        
        if distance < 0.18:
            return True

        return False

    def target_distance(self, robot_id = 0, target_id = 0):
        robot_position, _   = self.robots[robot_id].get_position_and_orientation()
        target_position, _  = self.pb_client.getBasePositionAndOrientation(self.targets[target_id])

        dif      = numpy.array(target_position) - numpy.array(robot_position)
        distance = (numpy.sum(dif**2))**0.5

        return distance

    def out_board(self, robot_id):
        robot_position, _   = self.robots[robot_id].get_position_and_orientation()

        if robot_position[2] < -1.0:
            return True

        return False


