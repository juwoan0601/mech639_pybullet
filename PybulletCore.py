""" Robotics Implimentation using pybullet
"""

import numpy as np
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

from math import *
from threading import Thread

import pybullet as p
import pybullet_data

from RoboticsCore import *

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

        self.endEffectorPosition = [0, 0, 0] ## base position
        self.endEffectorOrientation = p.getQuaternionFromEuler([0, 0, 0]) ## base orientation

        self.robotDt = self.dt

    def connectPybullet(self, robot_name = 'IndyRP2', joint_limit = True):
        '''
        #############################################################
        PYBULLET CONNECTION
        #############################################################
        '''

        ### Open GUI
        self.physicsClient = p.connect(p.GUI)

        # 기타 URDF 파일 경로 설정
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) 

        # 기존 번잡한 UI를 깔끔하게 바꿔줌
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

        ### Set perspective camera
        p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0, 0.7])

        p.setGravity(self.g_vector[0], self.g_vector[1], self.g_vector[2]) # set gravity

        ### Add objects from URDF
        self.planeId = p.loadURDF("plane.urdf")

        if robot_name == 'Indy7': # Indy7 (6DOF)
            self.robotId = p.loadURDF(self.__filepath + "/urdf/indy7/indy7.urdf",
                                      basePosition=self.startPosition, baseOrientation=self.startOrientation)
            self.M = np.array([
                [1, 0, 0,      0],
                [0, 1, 0, -0.187],
                [0, 0, 1, 1.3005],
                [0, 0, 0,      1]])
            self.Blist = np.array([
                [ +0.187,0,0,0, 0,1],
                [-1.0775,0,0,0,-1,0],
                [ -0.578,0,0,0,-1,0],
                [  0.183,0,0,0, 0,1],
                [ -0.228,0,0,0,-1,0],
                [      0,0,0,0, 0,1]]).T

        elif robot_name == 'IndyRP2': # IndyRP2 (7DOF)
            self.robotId = p.loadURDF(self.__filepath + "/urdf/indyRP2/indyrp2.urdf",
                                      basePosition=self.startPosition, baseOrientation=self.startOrientation)
            self.M = np.array([
                [ 1, 0, 0,       0],
                [ 0, 1, 0, -0.1864],
                [ 0, 0, 1,  1.2670],
                [ 0, 0, 0,       1]])
            self.Blist = np.array([
                [ 0.1864,0,0,0,	0,1],
                [-0.9675,0,0,0,-1,0],
                [-0.0073,0,0,0,	0,1],
                [ -0.518,0,0,0,-1,0],
                [  0.183,0,0,0,	0,1],
                [ -0.168,0,0,0,-1,0],
                [      0,0,0,0,	0,1]]).T
            self.Slist = np.array([
                [      0,0,0,0, 0,1],
                [ 0.2995,0,0,0,-1,0],
                [-0.1937,0,0,0, 0,1],
                [  0.749,0,0,0,-1,0],
                [-0.0034,0,0,0, 0,1],
                [  1.099,0,0,0,-1,0],
                [-0.1864,0,0,0, 0,1]]).T
            
        else:
            print(self.__BoldText + self.__RedText + "There are no available robot: {}".format(robot_name) + self.__DefaultText + self.__BlackText)
            return
        
        ### Reset robot's configuration
        self.endEffectorOffset = 0.065 # Distance From CoM of last link to End Effector
        self.numJoint = p.getNumJoints(self.robotId)
        if joint_limit == False:
            for idx in range(self.numJoint):
                p.changeDynamics(self.robotId, idx, jointLowerLimit=-314, jointUpperLimit=314)

        print(self.__BoldText + self.__BlueText + "****** LOAD SUCCESS ******" + self.__DefaultText + self.__BlackText)
        print(self.__BoldText + "Robot name" + self.__DefaultText + ": {}".format(robot_name))
        print(self.__BoldText + "DOF" + self.__DefaultText + ": {}".format(self.numJoint))
        print(self.__BoldText + "Joint limit" + self.__DefaultText + ": {}".format(joint_limit))

        ### Add objects
        # Generate Endeffector shape
        self.endEffectorSimulateId = p.createMultiBody(
            baseMass=0, 
            baseVisualShapeIndex=
                p.createVisualShape(p.GEOM_CYLINDER, length=0.03, radius=0.025, rgbaColor=[1, 0, 0, 0.5]))
        
        self.endEffectorExpectedId = p.createMultiBody(
            baseMass=0, 
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_CYLINDER, length=0.03, radius=0.05, rgbaColor=[0, 1, 0, 0.5]))
        
        # Generate Axis of world coordinate
        self.xAxisId = p.createMultiBody(
            baseMass=0, 
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_CYLINDER, length=2, radius=0.01, rgbaColor=[1, 0, 0, 1]))
        p.resetBasePositionAndOrientation(self.xAxisId, [1,0,0], [0,1,0,1])
        self.yAxisId = p.createMultiBody(
            baseMass=0,baseVisualShapeIndex=p.createVisualShape(p.GEOM_CYLINDER, length=2, radius=0.01, rgbaColor=[0, 1, 0, 1]))
        p.resetBasePositionAndOrientation(self.yAxisId, [0,1,0], [-1,0,0,1])
        self.zAxisId = p.createMultiBody(
            baseMass=0, 
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_CYLINDER, length=2, radius=0.01, rgbaColor=[0, 0, 1, 1]))
        p.resetBasePositionAndOrientation(self.zAxisId, [0,0,1], [0,0,1,0])

        # Add Debug Item - Axis of End-effector
        self.eexAxisId = p.createMultiBody(
            baseMass=0, 
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05,0.005,0.005], rgbaColor=[1, 0, 0, 1]))
        self.eeyAxisId = p.createMultiBody(
            baseMass=0, 
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.005,0.05,0.005], rgbaColor=[0, 1, 0, 1]))
        self.eezAxisId = p.createMultiBody(
            baseMass=0, 
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.005,0.005,0.05], rgbaColor=[0, 0, 1, 1]))

        # Add Debug Item - Joint Angle
        p.addUserDebugParameter("Set Joint Angle (deg)",1,0,1)
        self.jointAngleDebugParamId = np.zeros([self.numJoint],int)
        for i in range(self.numJoint):
            self.jointAngleDebugParamId[i] = p.addUserDebugParameter(f"Joint {i}",-180,180,0)

        # Add Debug Item - Circular Path Parameter
        self.debugMode = False
        p.addUserDebugParameter("Set Circle Parameter",1,0,1)
        self.circlePathDebugParamId = dict()
        self.circlePathDebugParamId['x'] = p.addUserDebugParameter("Location of center - X",-2,2,0)
        self.circlePathDebugParamId['y'] = p.addUserDebugParameter("Location of center - Y",-2,2,0)
        self.circlePathDebugParamId['z'] = p.addUserDebugParameter("Location of center - Z",0,2,0.5)
        self.circlePathDebugParamId['r'] = p.addUserDebugParameter("Radius",0,1,0.5)
        self.circlePathDebugParamId['alpha'] = p.addUserDebugParameter("Alpha (deg)",0,90,45)
        self.traceIds = []

        ### Start simulation
        self.configureDebugMode(self.debugMode)
        self._q_des = np.zeros([self.numJoint,1])
        self.__isSimulation = True
        self._thread = Thread(target=self._setRobotJoint)
        self._thread.start()

    def disconnectPybullet(self):
        '''
        #############################################################
        PYBULLET DISCONNECTION
        #############################################################
        '''

        self.__isSimulation = False
        p.disconnect()
        print(self.__BoldText + self.__BlueText + "Disconnect Success!" + self.__DefaultText + self.__BlackText)

    def _setRobotJoint(self):
        """Assign Joint for loaded robot using pybullet.setJointMotorControlArray """ 
        import time
        import numpy as np
        while(self.__isSimulation == True):
            _startT = time.time()
            if self.debugMode == True:
                for i in range(self.numJoint):
                    self._q_des[i] = p.readUserDebugParameter(self.jointAngleDebugParamId[i],self.physicsClient)*np.pi/180.0
            p.setJointMotorControlArray(bodyUniqueId=self.robotId,
                                        jointIndices=range(self.numJoint),
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=self._q_des)
            
            # Endeffector
            end_effector_offset = np.dot(np.array
                                            (p.getMatrixFromQuaternion(
                                            p.getLinkState(self.robotId, self.numJoint-1)[1])).reshape((3,3)),
                                        np.array([0,0,self.endEffectorOffset]))
            self.endEffectorPosition = np.array(p.getLinkState(self.robotId, self.numJoint-1)[0])+end_effector_offset
            self.endEffectorOrientation = p.getLinkState(self.robotId, self.numJoint-1)[1]
            self.endEffectorPose = np.r_[np.c_[np.array(p.getMatrixFromQuaternion(p.getLinkState(self.robotId, self.numJoint-1)[1])).reshape((3,3)),self.endEffectorPosition],np.array([0,0,0,1]).reshape(1,4)]
            p.resetBasePositionAndOrientation(self.endEffectorSimulateId,
                                            self.endEffectorPosition, 
                                            self.endEffectorOrientation)
            p.resetBasePositionAndOrientation(self.eexAxisId,
                                            self.endEffectorPosition, 
                                            self.endEffectorOrientation)
            p.resetBasePositionAndOrientation(self.eeyAxisId,
                                            self.endEffectorPosition, 
                                            self.endEffectorOrientation)
            p.resetBasePositionAndOrientation(self.eezAxisId,
                                            self.endEffectorPosition, 
                                            self.endEffectorOrientation)
            
            T = FKinBody(self.M, self.Blist, self._q_des)
            qw = np.sqrt(1+T[0,0]+T[1,1]+T[2,2])/2.0
            p.resetBasePositionAndOrientation(self.endEffectorExpectedId, 
                                              (np.dot(T,np.array([0,0,self.endEffectorOffset,1])))[0:3],
                                              [(T[2,1]-T[1,2])/(4*qw),(T[0,2]-T[2,0])/(4*qw),(T[1,0]-T[0,1])/(4*qw),qw])

            p.stepSimulation()
            time.sleep(self.dt)
            _endT = time.time()
            self.robotDt = _endT - _startT

    def configureDebugMode(self, enable):
        self.debugMode = enable
        if enable == False:        
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        elif enable == True:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)

    def createCirclePathFromEulerZYX(self,phi,theta,psi,pos,r,N=36,verbose=False,maxShape=20):
        """Create Circle Path on 3D space using transfrom matrix

        :param phi: Z axis angle of circle
        :param theta: Y axis angle of circle
        :param psi: X axis angle of circle
        :param pos: Location of center of circle. ex) [0,0,0]
        :param r: radius of cirlce
        :param N: (optional) number of step points

        :return: List of SE(3) Matrix
        """
        R = RotEulerZYX(phi,theta,psi)
        T = np.r_[np.c_[R,pos],np.array([0,0,0,1]).reshape(1,4)]

        t = np.arange(0,2*np.pi,2*np.pi/N)
        traj = []
        for tt in t : 
            position = T[0:3,3]+r*np.cos(tt)*T[0:3,0]+r*np.sin(tt)*T[0:3,1]
            traj.append(np.r_[np.c_[R,position],np.array([0,0,0,1]).reshape(1,4)])
        traj = np.array(traj)

        # Plot circle in 3D Space
        iShape = round(N/maxShape)
        for i in range(len(t)):
            if i%iShape == 0:
                exec(f"trace{i}_visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[0, {i}/len(t), 1, 0.5])")
                exec(f"trace{i}_visual_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=trace{i}_visual_shape)")
                exec(f"p.resetBasePositionAndOrientation(trace{i}_visual_id, [traj[{i},0,3],traj[{i},1,3],traj[{i},2,3]], [1,0,0,1])")

        if verbose:
            print("Pose of Circle: ", traj)

        return traj

    def createArbitaryCirclePath(self,duration,dt,verbose=False,maxShape=20,seed=-1):
        """Create Circle Path on 3D space using transfrom matrix

        :param phi: Z axis angle of circle
        :param theta: Y axis angle of circle
        :param psi: X axis angle of circle
        :param pos: Location of center of circle. ex) [0,0,0]
        :param r: radius of cirlce
        :param N: (optional) number of step points

        :return: List of SE(3) Matrix
        """
        if seed<0:pass
        else: random.seed(seed)

        phi = random.random()*(np.pi) - np.pi/2
        theta = random.random()*(np.pi) - np.pi/2
        psi = random.random()*(np.pi) - np.pi/2
        rand_xy = random.random()*0.2-0.1
        rand_z = random.random()*0.3+0.8
        pos = [rand_xy,rand_xy,rand_z]
        r = random.random()*0.2+0.1
        N = round(duration/dt)

        R = RotEulerZYX(phi,theta,psi)
        T = np.r_[np.c_[R,pos],np.array([0,0,0,1]).reshape(1,4)]

        t = np.arange(0,2*np.pi,2*np.pi/N)
        traj = []
        for tt in t : 
            position = T[0:3,3]+r*np.cos(tt)*T[0:3,0]+r*np.sin(tt)*T[0:3,1]
            traj.append(np.r_[np.c_[R,position],np.array([0,0,0,1]).reshape(1,4)])
        traj = np.array(traj)

        # Plot circle in 3D Space
        iShape = round(N/maxShape)
        for i in range(len(t)):
            if i%iShape == 0:
                exec(f"trace{i}_visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[0, {i}/len(t), 1, 0.5])")
                exec(f"trace{i}_visual_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=trace{i}_visual_shape)")
                exec(f"p.resetBasePositionAndOrientation(trace{i}_visual_id, [traj[{i},0,3],traj[{i},1,3],traj[{i},2,3]], [1,0,0,1])")

        if verbose:
            print("Pose of Circle: ", traj)

        return traj

    def createCirclePathFromVector(self,p0,p1,N=36,verbose=False,maxShape=20):
        # Generate Arbitary circle path
        t = np.arange(0,2*np.pi,2*np.pi/N)
        r = np.linalg.norm(p1)
        p2 = np.cross(p0,p1)
        p1 =  p1 / np.linalg.norm(p1)
        p2 =  p2 / np.linalg.norm(p2)
        p3 = np.cross(p1,p2)

        R = np.c_[p1,p2,p3]
        T = np.r_[np.c_[R,p0],np.array([0,0,0,1]).reshape(1,4)]

        t = np.arange(0,2*np.pi,2*np.pi/N)
        traj = []
        for tt in t : 
            position = T[0:3,3]+r*np.cos(tt)*T[0:3,0]+r*np.sin(tt)*T[0:3,1]
            traj.append(np.r_[np.c_[R,position],np.array([0,0,0,1]).reshape(1,4)])
        traj = np.array(traj)

        # Plot circle in 3D Space
        iShape = round(N/maxShape)
        for i in range(len(t)):
            if i%iShape == 0:
                exec(f"trace{i}_visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[0, {i}/len(t), 1, 0.5])")
                exec(f"trace{i}_visual_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=trace{i}_visual_shape)")
                exec(f"p.resetBasePositionAndOrientation(trace{i}_visual_id, [traj[{i},0,3],traj[{i},1,3],traj[{i},2,3]], [1,0,0,1])")

        if verbose:
            print("Pose of Circle: ", traj)
        
        return traj

    def removeObjects(self,N=200):
        """Clear Object except Robot, Axis and Endeffector
        """
        def __isTempObject(id):
            if id in [self.robotId, 
                      self.planeId, 
                      self.xAxisId, 
                      self.yAxisId, 
                      self.zAxisId, 
                      self.eexAxisId,
                      self.eeyAxisId,
                      self.eezAxisId,
                      self.endEffectorSimulateId, 
                      self.endEffectorExpectedId]: 
                return False
            else: return True
        with tqdm(total=N, desc="[Remove Objects] ") as pbar:
            for i in [x for x in range(0,N) if __isTempObject(x)]: 
                p.removeBody(i)
                pbar.update(1)

    def removeTraces(self):
        """Clear Object except Robot, Axis and Endeffector
        """

        for i in tqdm(self.traceIds, desc="[Remove Traces] "): 
            p.removeUserDebugItem(i)
        self.traceIds = []

    def _mapJointAngleConstraint(self,jointAngles):
        for i in range(self.numJoint):
            if jointAngles[i] < np.pi and jointAngles[i] > -np.pi:
                jointAngles[i] = jointAngles[i]
            else:
                jointAngles[i] = (jointAngles[i]+np.pi) % (np.pi*2) - np.pi
        return np.array(jointAngles)

    def moveRobotByJointAngle(self, jointAngles, verbose=False):
        """Move Robot using joint angle

        :param jointAngles: list angle list of joint's angle in radian
        """
        
        jointAngles = self._mapJointAngleConstraint(jointAngles)
        self._q_des = np.array(jointAngles).reshape([self.numJoint])

        if (verbose == True):
            print(self.__BoldText + self.__BlueText + "Set desired joint angle: " + self.__DefaultText + self.__BlackText, end='')
            print(self._q_des)

    def moveRobotByPose(self, T, verbose=False, maxiter=1000,seed=-1):
        """Move Robot using SE3 transform matrix

        :param T: 4x4 Transform matrix of end effector
        """
        if seed<0:pass
        else: np.random.seed(seed)

        eomg = 0.001
        ev = 0.0001
        Tee = np.array([
            [ 1, 0, 0,  0],
            [ 0, 1, 0,  0],
            [ 0, 0, 1, self.endEffectorOffset],
            [ 0, 0, 0,  1]])
        
        isSucc = False
        trial = 0
        error = 99999
        thetalist = np.zeros([self.numJoint])
        while (isSucc == False) and (trial < maxiter):
            thetalist0 = (np.pi-2*np.random.rand(self.numJoint)*np.pi) # random inital joint angle
            [thetalist_IK, isSucc] = IKinBody(self.Blist, self.M, np.dot(T,TransInv(Tee)), thetalist0, eomg, ev, maxiter)
            T_fk = FKinBody(self.M, self.Blist,thetalist_IK)
            _error = np.linalg.norm(T-np.dot(Tee,T_fk))
            if error > _error: 
                thetalist = thetalist_IK
                error = _error
            trial+=1
            if (verbose == True): 
                print(f"Trial #{trial} Error: ", np.linalg.norm(T-np.dot(Tee,T_fk)))
                print(f"Current Theta List: ", thetalist)

        thetalist = self._mapJointAngleConstraint(thetalist)
        self._q_des = np.array(thetalist).reshape([self.numJoint])

        if (verbose == True):
            print(self.__BoldText + self.__BlueText + "Set desired joint angle: " + self.__DefaultText + self.__BlackText, end='')
            print(self._q_des)

    def moveRobotByVelocityTime(self, v, T, frame="body", verbose=False, dt=0.015):
        """Move Robot using endeffector's velocity

        :param v: 6x1 desired velocity vector [v,w] of end effector
        :param T: duration of time(sec) that robot move desired velocity
        :param frame: (optional) frame of Given velocity. "world" for absolute world frame, "body" for endeffector frame. default="body"
        """
        
        currT = time.time()
        endT = time.time()+T
        while currT < endT:
            _setStartT = time.time()
            if frame == "body":
                J = JacobianBody(self.Blist,self._q_des)
            elif frame == "world":
                J = JacobianSpace(self.Slist,self._q_des)
            theta_dot = np.dot(np.linalg.pinv(J),v)
            # theta_dot = np.dot(np.dot(Jb.T,np.linalg.inv(np.eye(6)*0+np.dot(Jb,Jb.T))),v) # damped solution
            thetalist = self._q_des + theta_dot*dt
            thetalist = self._mapJointAngleConstraint(thetalist)
            self._q_des = np.array(thetalist).reshape([self.numJoint])
            time.sleep(self.dt)
            currT = time.time()

            # Plot trace
            end_effector_offset = np.dot(np.array
                                            (p.getMatrixFromQuaternion(
                                            p.getLinkState(self.robotId, self.numJoint-1)[1])).reshape((3,3)),
                                        np.array([0,0,self.endEffectorOffset]))
            pointId = p.addUserDebugPoints([np.array(p.getLinkState(self.robotId, self.numJoint-1)[0])+end_effector_offset],
                                    [[1,0,0]],pointSize=5,lifeTime=5)
            self.traceIds.append(pointId)

            if (verbose == True):
                print(self.__BoldText + self.__BlueText + "Set desired joint angle: " + self.__DefaultText + self.__BlackText, end='')
                print(self._q_des)
                print(self.__BoldText + self.__BlueText + "Measured Last Joint's Position: " + self.__DefaultText + self.__BlackText, end='')
                print(p.getLinkState(self.robotId, self.numJoint-1)[0])
            _setEndT = time.time()
            self.setDt = _setEndT-_setStartT
        
    def moveRobotPoint2Point(self,startT,endT,duration,verbose=False, trace=False,dt=0.015):
        """Move Robot from point to point striaght line

        :param v: 6x1 desired velocity vector [v,w] of end effector
        :param T: duration of time(sec) that robot move desired velocity
        :param frame: (optional) frame of Given velocity. "world" for absolute world frame, "body" for endeffector frame. default="body"
        """

        # generate mid point
        N = round(duration/dt)
        Rstart, pstart = TransToRp(startT)
        Rend, pend = TransToRp(endT)
        traj = [[None]] * (N+1)
        for i in range(N+1):
            s = i/N
            traj[i] \
            = np.r_[np.c_[np.dot(Rstart, \
            MatrixExp3(MatrixLog3(np.dot(np.array(Rstart).T,Rend)) * s)), \
                    s * np.array(pend) + (1 - s) * np.array(pstart)], \
                    [[0, 0, 0, 1]]]
        traj = np.array(traj)
        print(traj[-1])

        # for i in tqdm(range(N), desc=f"dT = {round(_dt,5)} / {round(self.robotDt,5)} / {round(self.setDt,5)}"):
        for i in range(N):
            dp = np.array([
                traj[(i+1)%(traj.shape[0]),0,3] - traj[i,0,3],
                traj[(i+1)%(traj.shape[0]),1,3] - traj[i,1,3],
                traj[(i+1)%(traj.shape[0]),2,3] - traj[i,2,3],
                0,0,0]).T
            V = dp
            J = JacobianSpace(self.Slist,self._q_des)
            theta_dot = np.dot(np.linalg.pinv(J),V)
            thetalist = self._q_des + theta_dot
                
            thetalist = self._mapJointAngleConstraint(thetalist)
            self._q_des = np.array(thetalist).reshape([self.numJoint])
            time.sleep(self.dt)

            # Plot trace
            if trace:
                end_effector_offset = np.dot(np.array
                                                (p.getMatrixFromQuaternion(
                                                p.getLinkState(self.robotId, self.numJoint-1)[1])).reshape((3,3)),
                                            np.array([0,0,self.endEffectorOffset]))
                pointId = p.addUserDebugPoints([np.array(p.getLinkState(self.robotId, self.numJoint-1)[0])+end_effector_offset],
                                        [[1,0,0]],pointSize=5,lifeTime=5)
                self.traceIds.append(pointId)

        if (verbose == True):
            print(self.__BoldText + self.__BlueText + "Set desired joint angle: " + self.__DefaultText + self.__BlackText, end='')
            print(self._q_des)
            print(self.__BoldText + self.__BlueText + "Measured Last Joint's Position: " + self.__DefaultText + self.__BlackText, end='')
            print(p.getLinkState(self.robotId, self.numJoint-1)[0])

    def moveRobotArbitaryCircle(self,duration,verbose=False,trace=False,dt=0.014,jointLimit=False,plot=False,seed=-1):
        """Move Robot from point to point striaght line

        :param v: 6x1 desired velocity vector [v,w] of end effector
        :param T: duration of time(sec) that robot move desired velocity
        :param frame: (optional) frame of Given velocity. "world" for absolute world frame, "body" for endeffector frame. default="body"
        """
        if seed<0:pass
        else: random.seed(seed)

        # generate mid point
        N = round(duration/dt)
        print(N, self.robotDt)
        phi = random.random()*(np.pi) - np.pi/2
        theta = random.random()*(np.pi) - np.pi/2
        psi = random.random()*(np.pi) - np.pi/2
        rand_xy = random.random()*0.2-0.1
        rand_z = random.random()*0.3+0.8
        r = random.random()*0.2+0.1
        traj = self.createCirclePathFromEulerZYX(phi,theta,psi,[rand_xy,rand_xy,rand_z],r,N=N, maxShape=10)

        # Set initial Position
        self.MoveRobotByPose(traj[0],verbose=verbose,maxiter=10)
        time.sleep(1)

        # History 
        history = {
            "time":[],
            "des_traj":[],
            "real_traj":[],
            "joint_angle":[],
            "joint_velocity":[],
            "ee_position":[],
            "ee_velocity":[],
            "jacobian":[]}

        # for i in tqdm(range(N), desc=f"dT = {round(_dt,5)} / {round(self.robotDt,5)} / {round(self.setDt,5)}"):
        _startT = time.time()
        for i in range(N):
            dp = np.array([
                traj[(i+1)%(traj.shape[0]),0,3] - traj[i,0,3],
                traj[(i+1)%(traj.shape[0]),1,3] - traj[i,1,3],
                traj[(i+1)%(traj.shape[0]),2,3] - traj[i,2,3],
                0,0,0]).T
            V = dp
            J = JacobianSpace(self.Slist,self._q_des)
            theta_dot = np.dot(np.linalg.pinv(J),V)
            # theta_dot = np.dot(np.dot(Jb.T,np.linalg.inv(np.eye(6)*0+np.dot(Jb,Jb.T))),v) # damped solution
            thetalist = self._q_des + theta_dot
            
            if jointLimit:
                thetalist = self._mapJointAngleConstraint(thetalist)
            self._q_des = np.array(thetalist).reshape([self.numJoint])
            time.sleep(self.dt)

            # mark trace
            if trace:
                end_effector_offset = np.dot(np.array
                                                (p.getMatrixFromQuaternion(
                                                p.getLinkState(self.robotId, self.numJoint-1)[1])).reshape((3,3)),
                                            np.array([0,0,self.endEffectorOffset]))
                pointId = p.addUserDebugPoints([np.array(p.getLinkState(self.robotId, self.numJoint-1)[0])+end_effector_offset],
                                        [[1,0,0]],pointSize=5,lifeTime=5)
                self.traceIds.append(pointId)

            # save log history
            history["time"].append(time.time()-_startT)
            history["des_traj"].append(np.array(traj[i]))
            history["real_traj"].append(np.array(self.endEffectorPose))
            history["joint_angle"].append(np.array(self._q_des))
            history["joint_velocity"].append(np.array(theta_dot))
            history["ee_position"].append(np.array(self.endEffectorPosition))
            history["ee_velocity"].append(np.array(V[0:3]))
            history["jacobian"].append(np.array(J))

        _endT = time.time()

        if (verbose == True):
            print(f"Tracking Time: {_endT-_startT}")

        return history

    def moveRobotByPoseNull(self, T, verbose=False,N=10,mag=3,jointLimit=False,maxiter=1000,seed=-1):
        """Move Robot using SE3 transform matrix

        :param T: 4x4 Transform matrix of end effector
        """
        if seed<0:pass
        else: np.random.seed(seed)

        eomg = 0.001
        ev = 0.0001
        Tee = np.array([
            [ 1, 0, 0,  0],
            [ 0, 1, 0,  0],
            [ 0, 0, 1, self.endEffectorOffset],
            [ 0, 0, 0,  1]])
        
        isSucc = False
        trial = 0
        error = 99999
        thetalist = np.zeros([self.numJoint])
        # Inverse Kinematics
        while (isSucc == False) and (trial < maxiter):
            thetalist0 = (np.pi-2*np.random.rand(self.numJoint)*np.pi) # random inital joint angle
            [thetalist_IK, isSucc] = IKinBody(self.Blist, self.M, np.dot(T,TransInv(Tee)), thetalist0, eomg, ev, maxiter)
            T_fk = FKinBody(self.M, self.Blist,thetalist_IK)
            _error = np.linalg.norm(T-np.dot(Tee,T_fk))
            if error > _error: 
                thetalist = thetalist_IK
                error = _error
            trial+=1
            if (verbose == True): 
                print(f"Trial #{trial} Error: ", np.linalg.norm(T-np.dot(T_fk,Tee)))
                print(f"Current Theta List: ", thetalist)

        # joint limit
        thetalist = self._mapJointAngleConstraint(thetalist)
        self._q_des = np.array(thetalist).reshape([self.numJoint])

        # Explore null space
        for i in range(N):
            # previous configuration
            J = JacobianSpace(self.Slist,self._q_des)
            mu = np.sqrt(np.linalg.det(np.dot(J,J.T)))
            
            # random null motion
            z = np.random.rand(self.numJoint).reshape([self.numJoint,1])*mag
            V = np.zeros(6).reshape([6,1])
            theta_dot = np.dot(np.linalg.pinv(J),V) + np.dot((np.eye(self.numJoint)-np.dot(np.linalg.pinv(J),J)),z)
            _thetalist = self._q_des + theta_dot.reshape([1,self.numJoint]) # null motion theta_dot

            # newest configuration
            _J = JacobianSpace(self.Slist,np.array(_thetalist).reshape([self.numJoint]))
            _mu = np.sqrt(np.linalg.det(np.dot(_J,_J.T)))
            print("mu: ",_mu)
            self._q_des = np.array(_thetalist).reshape([self.numJoint])

            # save maximum manipulability
            if mu < _mu:
                mu = _mu
                thetalist = _thetalist
            else: 
                thetalist = self._q_des
            
            # joint limit
            if jointLimit:
                thetalist = self._mapJointAngleConstraint(thetalist)
            time.sleep(0.2)

        print("Max mu: ", mu)
        self._q_des = np.array(thetalist).reshape([self.numJoint])
        time.sleep(0.2)

        if (verbose == True):
            print(self.__BoldText + self.__BlueText + "Set desired joint angle: " + self.__DefaultText + self.__BlackText, end='')
            print(self._q_des)

    def moveRobotTrajactory(self,traj,verbose=False,trace=False,jointLimit=False,plot=False,initMaxMani=False,seed=-1,nullMotion=False,potentialFunc=None,potentialK=1):
        """Move Robot from point to point striaght line

        :param v: 6x1 desired velocity vector [v,w] of end effector
        :param T: duration of time(sec) that robot move desired velocity
        :param frame: (optional) frame of Given velocity. "world" for absolute world frame, "body" for endeffector frame. default="body"
        """

        # Set initial Position
        if initMaxMani:
            self.moveRobotByPoseMaxManip(traj[0],miniter=100,seed=seed)
        else:
            self.moveRobotByPose(traj[0],verbose=verbose,maxiter=100,seed=seed)
        time.sleep(1)

        # History 
        history = {
            "time":[],
            "des_traj":[],
            "real_traj":[],
            "joint_angle":[],
            "joint_velocity":[],
            "ee_position":[],
            "ee_velocity":[],
            "jacobian":[],
            "potential_value":[]}

        _startT = time.time()
        pValue = 0

        for i in range(len(traj)):
            # get velocity value from trajectory
            dp = np.array([
                traj[(i+1)%(traj.shape[0]),0,3] - traj[i,0,3],
                traj[(i+1)%(traj.shape[0]),1,3] - traj[i,1,3],
                traj[(i+1)%(traj.shape[0]),2,3] - traj[i,2,3],
                0,0,0]).T
            Vp_s = dp
            Jsb_b = JacobianBody(self.Blist,self._q_des)
            Tsb = FKinBody(self.M,self.Blist,self._q_des)
            Vsb_b = np.r_[np.dot(TransInv(Tsb)[0:3,0:3],Vp_s[0:3]),[0,0,0]].reshape([6,1]) # 6x1 vector
            theta_dot = np.dot(np.linalg.pinv(Jsb_b),Vsb_b)

            # Additional Calculation (Not-used value)
            T = np.eye(4)
            for j in range(1, self.numJoint):
                T = np.dot(T, MatrixExp6(VecTose3(np.array(self.Slist)[:, j - 1] * self._q_des[j - 1])))
            AdjointG = Adjoint(T)
            Vsb_s = np.dot(AdjointG,Vsb_b)

            # calculate joint velocity using jacobian
            if potentialFunc is None:
                theta_dot = np.dot(np.linalg.pinv(Jsb_b),Vsb_b)
                pValue = 0
            elif potentialFunc == "MIN_SUM_ANGLE": # minimize sum of joint angle
                pValue = np.sum(self._q_des)
                if nullMotion:
                    z = -potentialK*(np.ones((self.numJoint,1))) # z = -k dp/dtheta, p = sum(theta1+...+thetaN)
                    theta_dot = np.dot(np.linalg.pinv(Jsb_b),Vsb_b) + np.dot((np.eye(self.numJoint)-np.dot(np.linalg.pinv(Jsb_b),Jsb_b)),z)
                else:
                    print("potentialFunc is Not None But nullMotion is True. No nullMotion")
            elif potentialFunc == "MIN_SQR_ANGLE": # minimize square sum of joint angle
                pValue = np.linalg.norm(np.array(self._q_des))
                if nullMotion:
                    z = -potentialK*(np.array(self._q_des).reshape([self.numJoint,1])) # z = -k dp/dtheta, p = sum(0.5*(theta1)^2+...+0.5*(thetaN)^2)
                    theta_dot = np.dot(np.linalg.pinv(Jsb_b),Vsb_b) + np.dot((np.eye(self.numJoint)-np.dot(np.linalg.pinv(Jsb_b),Jsb_b)),z)
                else:
                    print("potentialFunc is Not None But nullMotion is True. No nullMotion")

            # theta_dot = np.dot(np.dot(Jb.T,np.linalg.inv(np.eye(6)*0+np.dot(Jb,Jb.T))),v) # damped solution
            thetalist = np.array(self._q_des).reshape([self.numJoint,1]) + theta_dot

            # check joint limit
            if jointLimit:
                thetalist = self._mapJointAngleConstraint(thetalist)
            self._q_des = np.array(thetalist).reshape([self.numJoint])

            # check using forward kinematics
            T_cal = FKinBody(self.M,self.Blist,self._q_des)
            T_des = traj[(i+1)%(traj.shape[0])]

            # mark trace
            if trace:
                end_effector_offset = np.dot(np.array
                                                (p.getMatrixFromQuaternion(
                                                p.getLinkState(self.robotId, self.numJoint-1)[1])).reshape((3,3)),
                                            np.array([0,0,self.endEffectorOffset]))
                self.endEffectorPosition = np.array([np.array(p.getLinkState(self.robotId, self.numJoint-1)[0])+end_effector_offset]).reshape([3,1])
                self.endEffectorOrientation = p.getLinkState(self.robotId, self.numJoint-1)[1]
                self.endEffectorPose = np.r_[np.c_[np.array(p.getMatrixFromQuaternion(self.endEffectorOrientation)).reshape((3,3)),self.endEffectorPosition],np.array([0,0,0,1]).reshape(1,4)]
                pos_rms_error = np.linalg.norm(traj[i][0:3,3]-self.endEffectorPose[0:3,3])
                pointId = p.addUserDebugPoints([self.endEffectorPosition],
                                        [[pos_rms_error/0.1,1-pos_rms_error/0.1,0]],pointSize=5,lifeTime=5)
                self.traceIds.append(pointId)

            # save log history
            history["time"].append(time.time()-_startT)
            history["des_traj"].append(np.array(traj[i]))
            history["real_traj"].append(np.array(self.endEffectorPose))
            history["joint_angle"].append(np.squeeze(self._q_des))
            history["joint_velocity"].append(np.squeeze(theta_dot))
            history["ee_position"].append(np.squeeze(self.endEffectorPosition))
            history["ee_velocity"].append(np.squeeze(Vsb_b[0:3]))
            history["jacobian"].append(np.array(Jsb_b))
            history["potential_value"].append(pValue)

            # pause dt
            time.sleep(self.dt)

        _endT = time.time()

        if verbose:
            print(f"Tracking Time: {_endT-_startT}")
        
        if plot:
            self.plotTrackingResult(history)

        return history

    def moveRobotByPoseMaxManip(self, T, miniter=10, verbose=False, maxiter=1000, eomg=0.01,ev=0.001,seed=-1):
        """Move Robot using SE3 transform matrix

        :param T: 4x4 Transform matrix of end effector
        """
        if seed<0:pass
        else: np.random.seed(seed)

        Tee = np.array([
            [ 1, 0, 0,  0],
            [ 0, 1, 0,  0],
            [ 0, 0, 1, self.endEffectorOffset],
            [ 0, 0, 0,  1]])
        
        isSucc = False
        trial_ik = 0
        trial_pos = 0
        error = 99999
        mu = 0
        thetalist = np.zeros([self.numJoint])
        thetalist_fin = np.zeros([self.numJoint])
        while (trial_pos < miniter):
            while (isSucc == False) and (trial_ik < maxiter):
                thetalist0 = (np.pi-2*np.random.rand(self.numJoint)*np.pi) # random inital joint angle
                [thetalist_IK, isSucc] = IKinBody(self.Blist, self.M, np.dot(T,TransInv(Tee)), thetalist0, eomg, ev, maxiter,verbose=verbose)
                T_fk = FKinBody(self.M, self.Blist,thetalist_IK)
                _error = np.linalg.norm(T-np.dot(Tee,T_fk))
                if error > _error: 
                    thetalist = thetalist_IK
                    error = _error
                trial_ik+=1
                if (verbose == True): 
                    print(f"Trial #{trial_ik} Error: ", np.linalg.norm(T-np.dot(Tee,T_fk)))
                    print(f"Current Theta List: ", thetalist)
            isSucc = False
            trial_ik = 0

            J = JacobianSpace(self.Slist,thetalist)
            _mu = np.sqrt(np.linalg.det(np.dot(J,J.T)))
            if mu < _mu:
                mu = _mu
                thetalist_fin = thetalist
                print(f"mu update: {mu}")
            trial_pos+=1

        thetalist_fin = self._mapJointAngleConstraint(thetalist_fin)
        self._q_des = np.array(thetalist_fin).reshape([self.numJoint])

        if (verbose == True):
            print(self.__BoldText + self.__BlueText + "Set desired joint angle: " + self.__DefaultText + self.__BlackText, end='')
            print(self._q_des)

    def calManipuladility(self,J):
        J = np.array(J)
        return np.sqrt(np.linalg.det(np.dot(J,J.T)))
    
    def plotTrackingResult(self,data):
        his = data
        N = len(his["des_traj"])
        des_pos = []
        pos_rms_error = []
        pos_error = []
        mu = []
        eulerZYX = []
        eulerZYX_dot = []
        mag_ee_vel = []
        for i in range(N):
            pos_rms_error.append(np.linalg.norm(his["des_traj"][i][0:3,3]-his["real_traj"][i][0:3,3]))
            des_pos.append(his["des_traj"][i][0:3,3])
            pos_error.append(his["des_traj"][i][0:3,3]-his["real_traj"][i][0:3,3])
            mu.append(np.sqrt(np.linalg.det(np.dot(his["jacobian"][i],his["jacobian"][i].T))))
            mag_ee_vel.append(np.linalg.norm(his["ee_velocity"][i]))
            r1,r2=RotToEulerZYZ(his["des_traj"][i][0:3,0:3])
            eulerZYX.append(r1)
            phi = r1[0]
            theta = r1[1]
            Tr = np.array([
                [0,-np.sin(phi),np.cos(phi)*np.sin(theta)],
                [0,np.cos(phi),np.sin(phi)*np.sin(theta)],
                [1,0,np.cos(theta)]])
            repT = np.r_[np.c_[np.eye(3),np.zeros((3,3))],np.c_[np.zeros((3,3)),Tr]]
            Jr = np.dot(np.dot(np.linalg.pinv(repT),his["jacobian"][i]),his["joint_velocity"][i])
            eulerZYX_dot.append(Jr[3:6])

        plt.figure(figsize=(10,30))

        pN=11
        ax1 = plt.subplot(pN, 1, 1)
        plt.plot(his["time"], his["joint_angle"])
        maxA = np.max(np.abs( his["joint_angle"]))
        plt.title(f"Joint Angle(rad) | MAX: {round(maxA,4)}")
        plt.ylabel('Joint Angle(rad)')
        plt.legend([f"q{x}" for x in range(1,8)])
        plt.xticks(visible=False)

        ax2 = plt.subplot(pN, 1, 2, sharex=ax1)
        plt.plot(his["time"], his["joint_velocity"])
        maxV = np.max(np.abs(his["joint_velocity"]))
        stdV = np.std(his["joint_velocity"])
        plt.title(f"Joint Velocity(rad/s) | MAX:{round(maxV,4)}, STD:{round(stdV,4)}")
        plt.ylabel('Joint Velocity(rad/s)')
        plt.legend([f"q{x}" for x in range(1,8)])
        plt.xticks(visible=False)

        ax3 = plt.subplot(pN, 1, 3)
        plt.plot(his["time"], mu)
        plt.title(f"Manipulability | MIN: {round(np.min(mu),4)}, MAX:{round(np.max(mu),4)}")
        plt.ylabel('Manipulability')
        plt.xticks(visible=False)

        ax4 = plt.subplot(pN, 1, 4)
        plt.plot(his["time"], des_pos)
        plt.title('Desired Position')
        plt.ylabel('Desired [m]')
        plt.legend(["x","y","z"])
        plt.xticks(visible=False)

        ax5 = plt.subplot(pN, 1, 5)
        plt.plot(his["time"], his["ee_position"])
        plt.title('End Effector Position')
        plt.ylabel('End Effector Position [m]')
        plt.legend(["x","y","z"])
        plt.xticks(visible=False)

        ax6 = plt.subplot(pN, 1, 6)
        plt.plot(his["time"], his["ee_velocity"])
        plt.plot(his["time"], mag_ee_vel)
        totalT = his["time"][-1]-his["time"][0]
        plt.title(f"End Effector Velocity | {round(np.mean(mag_ee_vel),4)}m/s - Tracking Time: {round(totalT,2)} sec")
        plt.ylabel('End Effector Velocity [m/s]')
        plt.legend(["Vx","Vy","Vz", "|V|"])
        plt.xticks(visible=False)

        ax7 = plt.subplot(pN, 1, 7)
        plt.plot(his["time"], pos_error)
        plt.plot(his["time"], pos_rms_error)
        plt.title(f"End Effector Position Error: AVG: {round(np.mean(pos_rms_error),4)} m, MAX: {round(np.max(pos_rms_error),4)}")
        plt.ylabel('End Effector Position Error [m]')
        plt.legend(["dx","dy","dz","d"])
        plt.xticks(visible=False)

        ax8 = plt.subplot(pN, 1, 8)
        plt.plot(his["time"], eulerZYX)
        plt.title('End Effector Euler ZYX Angle')
        plt.ylabel('End Effector Euler ZYX Angle [rad]')
        plt.legend(["phi","theta","psi"])
        plt.xticks(visible=False)

        ax9 = plt.subplot(pN, 1, 9)
        plt.plot(his["time"], eulerZYX_dot)
        plt.title('End Effector Euler ZYX Angle Velocity')
        plt.ylabel('End Effector Euler ZYX Angle Velocity [rad/s]')
        plt.legend(["phi","theta","psi"])
        plt.xticks(visible=False)

        ax10 = plt.subplot(pN, 1, 10)
        plt.plot(his["time"], his["potential_value"])
        meanP = np.mean(his["potential_value"])
        maxP = np.max(his["potential_value"])
        plt.title(f"Potential Value | AVG: {round(meanP,4)}, MAX: {round(maxP,4)}")
        plt.ylabel('Potential Value')

        plt.grid(True)
        plt.xlabel("Time (sec)")
        plt.tight_layout()
        plt.show()
