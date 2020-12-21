import functools
import inspect
import pybullet
import numpy


class PybulletClient():
    """A wrapper for pybullet to manage different clients."""
    def __init__(self, render = False, view_camera_distance = 1.5, view_camera_angle = -50.0):
        
        print("creating pybullet client")

        if render:
            connection_mode=pybullet.GUI
        else:
            connection_mode=pybullet.DIRECT
            
        self._client = pybullet.connect(connection_mode, options='--background_color_red=0.128 --background_color_green=0.12 --background_color_blue=0.2457')

        self.__getattr__("resetDebugVisualizerCamera")(cameraDistance = view_camera_distance, cameraYaw=0.0, cameraPitch=view_camera_angle, cameraTargetPosition=[0, 0.0, 0])
        #self.__getattr__("resetDebugVisualizerCamera")(cameraDistance = 8.5, cameraYaw=0.0, cameraPitch=-89.9, cameraTargetPosition=[0, 0.0, 0])
        self.__getattr__("configureDebugVisualizer")(pybullet.COV_ENABLE_GUI,0)
        self.__getattr__("configureDebugVisualizer")(pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW,1)

        self._shapes = {}

    def __del__(self):
        try:
            pybullet.disconnect(physicsClientId=self._client)
        except pybullet.error:
            print("pybullet.disconnect error")

    def __getattr__(self, name):
        attribute = getattr(pybullet, name)
        if inspect.isbuiltin(attribute):
            attribute = functools.partial(attribute, physicsClientId=self._client)
        '''
        else:
            print("attr ERROR ", name)
        '''
        return attribute


    def get_image(self, yaw, pitch, roll, distance, target_x, target_y, target_z, width = 512, height = 512, fov = 120):


        vm = self.__getattr__("computeViewMatrixFromYawPitchRoll")([target_x, target_y, target_z], distance, yaw, pitch, roll, 2)


        pm = self.__getattr__("computeProjectionMatrixFOV")(fov=fov,
                                                       aspect=width / height,
                                                       nearVal=0.0001,
                                                       farVal=10.1) 

        w, h, rgb, deth, seg = self.__getattr__("getCameraImage")(width=width,
                                                             height=height,
                                                             viewMatrix=vm,
                                                             projectionMatrix=pm)
                                                             #renderer=self._client.ER_BULLET_HARDWARE_OPENGL)

        rgb = numpy.array(rgb)
        rgb = rgb[:, :, :3]
       
        return rgb