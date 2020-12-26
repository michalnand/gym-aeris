

class RobotBasic:
    def __init__(self, pb_client, urdf_file, x, y, z, initial_angle, left_wheel_joint = 1, right_wheel_joint = 2, inertia = 0.8, scale = 1.0):

        self.pb_client   = pb_client

        orientation      = self.pb_client.getQuaternionFromEuler([0, 0, initial_angle])
        self.pb_robot    = self.pb_client.loadURDF(urdf_file, [x, y, z], orientation, globalScaling=scale)

        self.left_wheel_joint   = left_wheel_joint
        self.right_wheel_joint  = right_wheel_joint
        self.inertia = inertia


    def get_position_and_orientation(self):
        return self.pb_client.getBasePositionAndOrientation(self.pb_robot)


    def get_wheel_position(self):
        l_pos, l_vel, l_react, l_torque = self.pb_client.getJointState(self.pb_robot, self.left_wheel_joint)
        r_pos, r_vel, r_react, r_torque = self.pb_client.getJointState(self.pb_robot, self.right_wheel_joint)
        return l_pos, r_pos

    def get_wheel_torque(self):
        l_pos, l_vel, l_react, l_torque = self.pb_client.getJointState(self.pb_robot, self.left_wheel_joint)
        r_pos, r_vel, r_react, r_torque = self.pb_client.getJointState(self.pb_robot, self.right_wheel_joint)
        return l_torque, r_torque

    def get_wheel_velocity(self):
        l_pos, l_vel, l_react, l_torque = self.pb_client.getJointState(self.pb_robot, self.left_wheel_joint)
        r_pos, r_vel, r_react, r_torque = self.pb_client.getJointState(self.pb_robot, self.right_wheel_joint)
        return l_vel, r_vel

    def get_position(self):
        position, orientation = self.pb_client.getBasePositionAndOrientation(self.pb_robot)
        
        x, y, z = position
        
        orientation = self.pb_client.getEulerFromQuaternion(orientation)
        pitch, roll, yaw = orientation

        return x, y, z, pitch, roll, yaw


    def set_throttle(self, left_power, right_power):
        vl, vr = self.get_wheel_velocity()

        vl  =  self.inertia*vl  + (1.0 - self.inertia)*left_power
        vr  =  self.inertia*vr  + (1.0 - self.inertia)*right_power

        self.set_velocity(vl, vr)
   
    def set_velocity(self, left_velocity, right_velocity):
        self._set_wheel_velocity(left_velocity, right_velocity)


    def _set_wheel_velocity(self, left_velocity, right_velocity):

        self.pb_client.setJointMotorControl2(self.pb_robot,
                                             jointIndex =   self.left_wheel_joint,
                                             controlMode=   self.pb_client.VELOCITY_CONTROL,
                                             targetVelocity      =   left_velocity)

        self.pb_client.setJointMotorControl2(self.pb_robot,
                                             jointIndex =   self.right_wheel_joint,
                                             controlMode=   self.pb_client.VELOCITY_CONTROL,
                                             targetVelocity      =   right_velocity)
   