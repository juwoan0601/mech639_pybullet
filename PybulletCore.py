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

    def connect_pybullet(self, robot_name = 'IndyRP2', joint_limit = True):
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
        self._q_des = np.zeros([self.numJoint])
        self.__isSimulation = True
        self._thread = Thread(target=self._SetRobotJoint)
        self._thread.start()

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
        """Assign Joint for loaded robot using pybullet.setJointMotorControlArray """ 
        import time
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

    def CreateCirclePathFromEulerZYX(self,phi,theta,psi,pos,r,N=36,verbose=False,maxShape=20):
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

    def CreateArbitaryCirclePath(self,duration,dt,verbose=False,maxShape=20):
        """Create Circle Path on 3D space using transfrom matrix

        :param phi: Z axis angle of circle
        :param theta: Y axis angle of circle
        :param psi: X axis angle of circle
        :param pos: Location of center of circle. ex) [0,0,0]
        :param r: radius of cirlce
        :param N: (optional) number of step points

        :return: List of SE(3) Matrix
        """
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

    def CreateCirclePathFromVector(self,p0,p1,N=36,verbose=False,maxShape=20):
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

    def RemoveObjects(self,N=200):
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

    def RemoveTraces(self):
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

    def MoveRobotByJointAngle(self, jointAngles, verbose=False):
        """Move Robot using joint angle

        :param jointAngles: list angle list of joint's angle in radian
        """
        
        jointAngles = self._mapJointAngleConstraint(jointAngles)
        self._q_des = np.array(jointAngles).reshape([self.numJoint])

        if (verbose == True):
            print(self.__BoldText + self.__BlueText + "Set desired joint angle: " + self.__DefaultText + self.__BlackText, end='')
            print(self._q_des)

    def MoveRobotByPose(self, T, verbose=False, maxiter=1000):
        """Move Robot using SE3 transform matrix

        :param T: 4x4 Transform matrix of end effector
        """
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
            [thetalist_IK, isSucc] = IKinBody(self.Blist, self.M, np.dot(TransInv(Tee),T), thetalist0, eomg, ev, maxiter)
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

    def MoveRobotByVelocityTime(self, v, T, frame="body", verbose=False, dt=0.015):
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
        
    def MoveRobotPoint2Point(self,startT,endT,duration,verbose=False, trace=False,dt=0.015):
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

    def circlingRobot(self,duration,verbose=False,trace=False,dt=0.014,jointLimit=False,plot=False):
        """Move Robot from point to point striaght line

        :param v: 6x1 desired velocity vector [v,w] of end effector
        :param T: duration of time(sec) that robot move desired velocity
        :param frame: (optional) frame of Given velocity. "world" for absolute world frame, "body" for endeffector frame. default="body"
        """
        # generate mid point
        N = round(duration/dt)
        print(N, self.robotDt)
        phi = random.random()*(np.pi) - np.pi/2
        theta = random.random()*(np.pi) - np.pi/2
        psi = random.random()*(np.pi) - np.pi/2
        rand_xy = random.random()*0.2-0.1
        rand_z = random.random()*0.3+0.8
        r = random.random()*0.2+0.1
        traj = self.CreateCirclePathFromEulerZYX(phi,theta,psi,[rand_xy,rand_xy,rand_z],r,N=N, maxShape=10)

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

    def MoveRobotByPoseNull(self, T, verbose=False,N=10,mag=3,jointLimit=False,maxiter=1000):
        """Move Robot using SE3 transform matrix

        :param T: 4x4 Transform matrix of end effector
        """
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
            [thetalist_IK, isSucc] = IKinBody(self.Blist, self.M, np.dot(TransInv(Tee),T), thetalist0, eomg, ev, maxiter)
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

        for i in range(N):
            J = JacobianSpace(self.Slist,self._q_des)
            mu = np.sqrt(np.linalg.det(np.dot(J,J.T)))

            z = np.random.rand(self.numJoint).reshape([self.numJoint,1])*mag
            V = np.zeros(6).reshape([6,1])
            theta_dot = np.dot(np.linalg.pinv(J),V) + np.dot((np.eye(self.numJoint)-np.dot(np.linalg.pinv(J),J)),z)
            _thetalist = self._q_des + theta_dot.reshape([1,self.numJoint])

            _J = JacobianSpace(self.Slist,np.array(_thetalist).reshape([self.numJoint]))
            _mu = np.sqrt(np.linalg.det(np.dot(_J,_J.T)))
            print("mu: ",_mu, mu)

            if mu < _mu:
                mu = _mu
                thetalist = _thetalist
            else: 
                thetalist = self._q_des
            
            if jointLimit:
                thetalist = self._mapJointAngleConstraint(thetalist)
            # _mu = 0
            # thetalist = self._q_des + theta_dot.reshape([1,self.numJoint])
            self._q_des = np.array(thetalist).reshape([self.numJoint])
            time.sleep(0.2)

        if (verbose == True):
            print(self.__BoldText + self.__BlueText + "Set desired joint angle: " + self.__DefaultText + self.__BlackText, end='')
            print(self._q_des)

    def MoveRobotTrajactory(self,traj,verbose=False,trace=False,jointLimit=False,plot=False,initMaxMani=False,Nfit=0):
        """Move Robot from point to point striaght line

        :param v: 6x1 desired velocity vector [v,w] of end effector
        :param T: duration of time(sec) that robot move desired velocity
        :param frame: (optional) frame of Given velocity. "world" for absolute world frame, "body" for endeffector frame. default="body"
        """

        # Set initial Position
        if initMaxMani:
            self.MoveRobotByPoseMaxManip(traj[0],miniter=100)
        else:
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
        N = len(traj)
        for i in range(N):
            if Nfit != 0:
                if i%round(N/Nfit) == 0:
                    self.MoveRobotByPoseMaxManip(traj[i])
                    continue
            dp = np.array([
                traj[(i+1)%(traj.shape[0]),0,3] - traj[i,0,3],
                traj[(i+1)%(traj.shape[0]),1,3] - traj[i,1,3],
                traj[(i+1)%(traj.shape[0]),2,3] - traj[i,2,3],
                0,0,0]).T
            V = dp
            J = JacobianSpace(self.Slist,self._q_des)
            theta_dot = np.dot(np.linalg.pinv(J),V)
            # theta_dot = np.dot(np.linalg.pinv(J),V) + (np.eye(self.numJoint)-np.dot(np.linalg.pinv(J),J))z
            # theta_dot = np.dot(np.dot(Jb.T,np.linalg.inv(np.eye(6)*0+np.dot(Jb,Jb.T))),v) # damped solution
            thetalist = self._q_des + theta_dot
            
            if jointLimit:
                thetalist = self._mapJointAngleConstraint(thetalist)
            self._q_des = np.array(thetalist).reshape([self.numJoint])
            time.sleep(self.dt)

            # mark trace
            if trace:
                pos_rms_error = np.linalg.norm(np.array(traj[i])[0:3,3]-np.array(self.endEffectorPose)[0:3,3])
                end_effector_offset = np.dot(np.array
                                                (p.getMatrixFromQuaternion(
                                                p.getLinkState(self.robotId, self.numJoint-1)[1])).reshape((3,3)),
                                            np.array([0,0,self.endEffectorOffset]))
                pointId = p.addUserDebugPoints([np.array(p.getLinkState(self.robotId, self.numJoint-1)[0])+end_effector_offset],
                                        [[pos_rms_error/0.1,1-pos_rms_error/0.1,0]],pointSize=5,lifeTime=5)
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

    def MoveRobotByPoseMaxManip(self, T, miniter=10, verbose=False, maxiter=1000, eomg=0.01,ev=0.001):
        """Move Robot using SE3 transform matrix

        :param T: 4x4 Transform matrix of end effector
        """
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