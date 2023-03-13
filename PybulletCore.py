
import numpy as np
import os
import time
import matplotlib.pyplot as plt

from math import *
from threading import Thread

import pybullet as p
import pybullet_data

class pybullet_core:

    def __init__(self):
        '''
        #############################################################
        PYBULLET INITIALIZATION
        #############################################################
        '''
        ###
        self.__WhiteText = "\033[37m"
        self.__BlackText = "\033[30m"
        self.__RedText = "\033[31m"
        self.__BlueText = "\033[34m"

        self.__DefaultText = "\033[0m"
        self.__BoldText = "\033[1m"
        ### Simulator configuration

        self.__filepath = os.getcwd()

        self.startPosition = [0, 0, 0] ## base position
        self.startOrientation = p.getQuaternionFromEuler([0, 0, 0]) ## base orientation

        self.g_vector = np.array([0, 0, -9.81]).reshape([3, 1])

        self.dt = 1. / 240  # Simulation Frequency
        # self.dt = 1./1000  # Simulation Frequency



    def connect_pybullet(self, robot_name = 'IndyRP2', joint_limit = True):
        '''
        #############################################################
        PYBULLET CONNECTION
        #############################################################
        '''

        ### Open GUI
        self.physicsClient = p.connect(p.GUI)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

        ### Set perspective camera
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0, 0])

        p.setGravity(self.g_vector[0], self.g_vector[1], self.g_vector[2]) # set gravity

        ### Add objects from URDF
        self.planeId = p.loadURDF("plane.urdf")

        if robot_name == 'Indy7': # Indy7 (6DOF)

            self.robotId = p.loadURDF(self.__filepath + "/urdf/indy7/indy7.urdf",
                                      basePosition=self.startPosition, baseOrientation=self.startOrientation)

        elif robot_name == 'IndyRP2': # IndyRP2 (7DOF)

            self.robotId = p.loadURDF(self.__filepath + "/urdf/indyRP2/indyrp2.urdf",
                                      basePosition=self.startPosition, baseOrientation=self.startOrientation)

        else:
            print(self.__BoldText + self.__RedText + "There are no available robot: {}".format(robot_name) + self.__DefaultText + self.__BlackText)
            return

        ### Reset robot's configuration
        self.numJoint = p.getNumJoints(self.robotId)
        if joint_limit == False:
            for idx in range(self.numJoint):
                p.changeDynamics(self.robotId, idx, jointLowerLimit=-314, jointUpperLimit=314)

        print(self.__BoldText + self.__BlueText + "****** LOAD SUCCESS ******" + self.__DefaultText + self.__BlackText)
        print(self.__BoldText + "Robot name" + self.__DefaultText + ": {}".format(robot_name))
        print(self.__BoldText + "DOF" + self.__DefaultText + ": {}".format(self.numJoint))
        print(self.__BoldText + "Joint limit" + self.__DefaultText + ": {}".format(joint_limit))


        ### Start simulation
        self._q_des = np.zeros([self.numJoint])
        self.__isSimulation = True
        self._thread = Thread(target=self._SetRobotJoint)
        self._thread.start()

        ### Add objects
        self.visualShapeId_1 = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 0, 0, 0.5])
        self.visualShapeId_2 = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.005, rgbaColor=[0, 0, 1, 0.3])




    def disconnect_pybullet(self):
        '''
        #############################################################
        PYBULLET DISCONNECTION
        #############################################################
        '''

        self.__isSimulation = False
        p.disconnect()
        print(self.__BoldText + self.__BlueText + "Disconnect Success!" + self.__DefaultText + self.__BlackText)




    def _SetRobotJoint(self):

        while(self.__isSimulation == True):
            p.setJointMotorControlArray(bodyUniqueId=self.robotId,
                                        jointIndices=range(self.numJoint),
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=self._q_des)

            p.stepSimulation()
            time.sleep(self.dt)



    def MoveRobot(self, angle, verbose=False):

        self._q_des = np.array(angle).reshape([self.numJoint])

        if (verbose == True):
            print(self.__BoldText + self.__BlueText + "Set desired joint angle: " + self.__DefaultText + self.__BlackText, end='')
            print(self._q_des)