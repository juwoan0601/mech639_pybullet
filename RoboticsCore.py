"""This core was re-writed core of `Mordern Robotics Github <https://github.com/NxRLab/ModernRobotics>`_ with different notation.
"""

import numpy as np
from tqdm import tqdm

def NearZero(z, threshold=1e-6):
    """Determines whether a scalar is small enough to be treated as zero

    :param z: A scalar input to check
    :param threshold: (Optional) criteria for zero. Default=1e-6
    :return: True if z is close to zero, false otherwise

    Example Input:
        z = -1e-7
    Output:
        True
    """
    return abs(z) < threshold

def Normalize(V):
    """Normalizes a vector

    :param V: A vector
    :return: A unit vector pointing in the same direction as z

    Example Input:
        V = np.array([1, 2, 3])
    Output:
        np.array([0.26726124, 0.53452248, 0.80178373])
    """
    return V / np.linalg.norm(V)

def RotX(psi):
    """Return Rotation Matrix along X-Axis

    :param psi: Rotation Angle
    :return: 3x3 Rotation Matrix

    Example Input:
        R = RotX(3.1415/2) # 90 degree
    Output:
        np.array([[1, 0, 0],
                  [0, 0,-1],
                  [0, 1, 0]])
    """
    return np.array([
        [1,          0,           0],
        [0,np.cos(psi),-np.sin(psi)],
        [0,np.sin(psi), np.cos(psi)]])

def RotY(theta):
    """Return Rotation Matrix along Y-Axis

    :param theta: Rotation Angle
    :return: 3x3 Rotation Matrix

    Example Input:
        R = RotY(3.1415/2)) # 90 degree
    Output:
        np.array([[ 0, 0, 1],
                  [ 0, 1, 0],
                  [-1, 0, 0]])
    """
    return np.array([
        [ np.cos(theta),0,np.sin(theta)],
        [             0,1,            0],
        [-np.sin(theta),0,np.cos(theta)]])

def RotZ(phi):
    """Return Rotation Matrix along Z-Axis

    :param theta: Rotation Angle
    :return: 3x3 Rotation Matrix

    Example Input:
        R = RotZ(3.1415/2)) # 90 degree
    Output:
        np.array([[0,-1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    """
    return np.array([
        [np.cos(phi),-np.sin(phi),0],
        [np.sin(phi), np.cos(phi),0],
        [          0,           0,1]])

def RotEulerZYX(phi,theta,psi):
    return np.dot(RotZ(phi),RotY(theta),RotX(psi))

def RotEulerZYZ(phi,theta,psi):
    return np.dot(RotZ(phi),RotY(theta),RotZ(psi))

def RotToEulerZYX(R):
    """Return Rotation Matrix along Z-Axis

    :param theta: Rotation Angle
    :return: 3x3 Rotation Matrix

    Example Input:
        R = [
            [0.5,   -0.1464, 0.8536],
            [0.5,    0.8536,-0.1464],
            [-0.7071,0.5,       0.5]]
        RotToEulerZYX(R)
    Output:
        np.array([0.78539816,
                  0.78539816,
                  0.78539816]) # pi/4
    """
    theta1 = -np.arcsin(R[2,0])
    theta2 = np.pi-theta1
    if R[2,0] == 1: # theta = p1/2
        phi1 = 0
        psi1 = phi1 + np.arctan2(R[0,1],R[0,2])
    elif R[2,0] == -1: # theta = -p1/2
        phi1 = 0
        psi1 = -phi1 + np.arctan2(-R[0,1],-R[0,2])
    else:
        psi1 = np.arctan2(R[2,1]/np.cos(theta1),R[2,2]/np.cos(theta1))
        phi1 = np.arctan2(R[1,0]/np.cos(theta1),R[0,0]/np.cos(theta1))
        psi2 = np.arctan2(R[2,1]/np.cos(theta2),R[2,2]/np.cos(theta2))
        phi2 = np.arctan2(R[1,0]/np.cos(theta2),R[0,0]/np.cos(theta2))
    return np.array([phi1,theta1,psi1])

def RotToEulerZYZ(R):
    """Return Rotation Matrix along Z-Axis

    :param theta: Rotation Angle
    :return: 3x3 Rotation Matrix
    """
    theta1 = -np.arccos(R[2,2])
    theta2 = np.pi-theta1
    if R[2,2] == -1: # theta = p1
        phi1 = 0
        psi1 = phi1 + np.arctan2(R[0,1],R[0,2])
    elif R[2,2] == 1: # theta = -p1
        phi1 = 0
        psi1 = -phi1 + np.arctan2(-R[0,1],-R[0,2])
    else:
        psi1 = np.arctan2(R[2,1]/np.sin(theta1),-R[2,0]/np.sin(theta1))
        phi1 = np.arctan2(R[1,2]/np.sin(theta1),R[0,2]/np.sin(theta1))
        psi2 = np.arctan2(R[2,1]/np.sin(theta2),-R[2,0]/np.sin(theta2))
        phi2 = np.arctan2(R[1,2]/np.sin(theta2),R[0,2]/np.sin(theta2))
    return np.array([phi1,theta1,psi1]), np.array([phi2,theta2,psi2])

def RotInv(R):
    """Inverts a rotation matrix

    :param R: A rotation matrix
    :return: The inverse of R

    Example Input:
        R = np.array([[0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0]])
    Output:
        np.array([[0, 1, 0],
                  [0, 0, 1],
                  [1, 0, 0]])
    """
    return np.array(R).T

def VecToso3(omg):
    """Converts a 3-vector to an so(3) representation

    :param omg: A 3-vector
    :return: The skew symmetric representation of omg

    Example Input:
        omg = np.array([1, 2, 3])
    Output:
        np.array([[ 0, -3,  2],
                  [ 3,  0, -1],
                  [-2,  1,  0]])
    """
    return np.array([[0,      -omg[2],  omg[1]],
                     [omg[2],       0, -omg[0]],
                     [-omg[1], omg[0],       0]])

def so3ToVec(so3mat):
    """Converts an so(3) representation to a 3-vector

    :param so3mat: A 3x3 skew-symmetric matrix
    :return: The 3-vector corresponding to so3mat

    Example Input:
        so3mat = np.array([[ 0, -3,  2],
                           [ 3,  0, -1],
                           [-2,  1,  0]])
    Output:
        np.array([1, 2, 3])
    """
    return np.array([so3mat[2][1], so3mat[0][2], so3mat[1][0]])

def AxisAng3(expc3):
    """Converts a 3-vector of exponential coordinates for rotation into axis-angle form

    :param expc3: A 3-vector of exponential coordinates for rotation
    :return omghat: A unit rotation axis
    :return theta: The corresponding rotation angle θ

    Example Input:
        expc3 = np.array([1, 2, 3])
    Output:
        (np.array([0.26726124, 0.53452248, 0.80178373]), 3.7416573867739413)
    """
    return (Normalize(expc3), np.linalg.norm(expc3))

def MatrixExp3(so3mat):
    """Computes the matrix exponential of a matrix in so(3) using `Rodrigues' Rotation formula <https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula>`_

    :param so3mat: A 3x3 skew-symmetric matrix
    :return: The matrix exponential of so3mat

    Example Input:
        so3mat = np.array([[ 0, -3,  2],
                           [ 3,  0, -1],
                           [-2,  1,  0]])
    Output:
        np.array([[-0.69492056,  0.71352099,  0.08929286],
                  [-0.19200697, -0.30378504,  0.93319235],
                  [ 0.69297817,  0.6313497 ,  0.34810748]])
    """
    omgtheta = so3ToVec(so3mat)
    if NearZero(np.linalg.norm(omgtheta)):
        return np.eye(3)
    else:
        theta = AxisAng3(omgtheta)[1]
        omgmat = so3mat / theta
        return np.eye(3) + np.sin(theta) * omgmat \
               + (1 - np.cos(theta)) * np.dot(omgmat, omgmat)
    
def MatrixLog3(R):
    """Computes the matrix logarithm of a rotation matrix

    :param R: A 3x3 rotation matrix
    :return: The matrix logarithm of R (wθ where R=e^wθ)

    Example Input:
        R = np.array([[0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0]])
    Output:
        np.array([[          0, -1.20919958,  1.20919958],
                  [ 1.20919958,           0, -1.20919958],
                  [-1.20919958,  1.20919958,           0]])
    """
    acosinput = (np.trace(R) - 1) / 2.0
    if acosinput >= 1:
        return np.zeros((3, 3))
    elif acosinput <= -1:
        if not NearZero(1 + R[2][2]):
            omg = (1.0 / np.sqrt(2 * (1 + R[2][2]))) \
                  * np.array([R[0][2], R[1][2], 1 + R[2][2]])
        elif not NearZero(1 + R[1][1]):
            omg = (1.0 / np.sqrt(2 * (1 + R[1][1]))) \
                  * np.array([R[0][1], 1 + R[1][1], R[2][1]])
        else:
            omg = (1.0 / np.sqrt(2 * (1 + R[0][0]))) \
                  * np.array([1 + R[0][0], R[1][0], R[2][0]])
        return VecToso3(np.pi * omg)
    else:
        theta = np.arccos(acosinput)
        return theta / 2.0 / np.sin(theta) * (R - np.array(R).T)

def RpToTrans(R, p):
    """Converts a rotation matrix and a position vector into homogeneous transformation matrix

    :param R: A 3x3 rotation matrix
    :param p: A 3-vector
    :return: A homogeneous transformation matrix corresponding to the inputs

    Example Input:
        R = np.array([[1, 0,  0],
                      [0, 0, -1],
                      [0, 1,  0]])
        p = np.array([1, 2, 5])
    Output:
        np.array([[1, 0,  0, 1],
                  [0, 0, -1, 2],
                  [0, 1,  0, 5],
                  [0, 0,  0, 1]])
    """
    return np.r_[np.c_[R, p], [[0, 0, 0, 1]]]

def TransToRp(T):
    """Converts a homogeneous transformation matrix(T) into a rotation matrix(R) and position vector(p)

    :param T: A homogeneous transformation matrix T = [R p; 0 1]
    :return R: The corresponding rotation matrix, R
    :return p: The corresponding position vector, p

    Example Input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        (np.array([[1, 0,  0],
                   [0, 0, -1],
                   [0, 1,  0]]),
         np.array([0, 0, 3]))
    """
    T = np.array(T)
    return T[0: 3, 0: 3], T[0: 3, 3]

def TransInv(T):
    """Inverts a homogeneous transformation matrix(T-1) using Equation: T^(-1) = [R^T -R^T*p; 0 1]

    :param T: A homogeneous transformation matrix
    :return: The inverse of T
    Uses the structure of transformation matrices to avoid taking a matrix inverse, for efficiency.

    Example input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        np.array([[1,  0, 0,  0],
                  [0,  0, 1, -3],
                  [0, -1, 0,  0],
                  [0,  0, 0,  1]])
    """
    R, p = TransToRp(T)
    Rt = np.array(R).T
    return np.r_[np.c_[Rt, -np.dot(Rt, p)], [[0, 0, 0, 1]]]

def VecTose3(V):
    """Converts a spatial velocity vector into a 4x4 matrix in se3

    :param V: A 6-vector representing a spatial velocity [v w]
    :return: The 4x4 se3 representation of V = [w_hat v; 0 0]

    Example Input:
        V = np.array([1, 2, 3, 4, 5, 6])
    Output:
        np.array([[ 0, -6,  5, 1],
                  [ 6,  0, -4, 2],
                  [-5,  4,  0, 3],
                  [ 0,  0,  0, 0]])
    """
    return np.r_[np.c_[VecToso3([V[3], V[4], V[5]]), [V[0], V[1], V[2]]],
                 np.zeros((1, 4))]

def se3ToVec(se3mat):
    """ Converts an se3 matrix ([w_hat v; 0 0]) into a spatial velocity vector [v w].

    :param se3mat: A 4x4 matrix in se3
    :return: The spatial velocity 6-vector corresponding to se3mat

    Example Input:
        se3mat = np.array([[ 0, -6,  5, 1],
                           [ 6,  0, -4, 2],
                           [-5,  4,  0, 3],
                           [ 0,  0,  0, 0]])
    Output:
        np.array([1, 2, 3, 4, 5, 6])
    """
    return np.r_[[se3mat[0][3], se3mat[1][3], se3mat[2][3]],
                 [se3mat[2][1], se3mat[0][2], se3mat[1][0]]]

def Adjoint(T):
    """ Computes the adjoint representation of a homogeneous transformation matrix

    :param T: A homogeneous transformation matrix
    :return: The 6x6 adjoint representation [AdT] of T = [R P_hat*R; 0 R].

    Example Input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        np.array([[1, 0,  0, 0, 0,  3],
                  [0, 0, -1, 3, 0,  0],
                  [0, 1,  0, 0, 0,  0],
                  [0, 0,  0, 1, 0,  0],
                  [0, 0,  0, 0, 0, -1],
                  [0, 0,  0, 0, 1,  0]])
                  
    """

    R, p = TransToRp(T)
    return np.r_[np.c_[R, np.dot(VecToso3(p), R)], np.c_[np.zeros((3, 3)), R]]

def ScrewToAxis(q, s, h):
    """Takes a parametric description of a screw axis and converts it to a normalized screw axis

    :param q: A point lying on the screw axis
    :param s: A unit vector in the direction of the screw axis (e.g. norm(w))
    :param h: The pitch of the screw axis
    :return: A normalized screw axis described by the inputs

    Example Input:
        q = np.array([3, 0, 0])
        s = np.array([0, 0, 1])
        h = 2
    Output:
        np.array([ 0, -3, 2, 0, 0, 1])
    """
    return np.r_[np.cross(q, s) + np.dot(h, s), s]

def AxisAng6(expc6):
    """Converts a 6-vector of exponential coordinates into screw axis-angle form

    :param expc6: A 6-vector of exponential coordinates for rigid-body motion S*θ = [v w]*θ
    :return S: The corresponding normalized screw axis
    :return theta: The distance traveled along/about S

    Example Input:
        expc6 = np.array([1, 2, 3, 1, 0, 0])
    Output:
        (np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0]), 1.0)
    """
    theta = np.linalg.norm([expc6[3], expc6[4], expc6[5]]) # it means the motion contains rotation.
    if NearZero(theta): # it means the motion is pure-translation (prismatic joint)
        theta = np.linalg.norm([expc6[3], expc6[4], expc6[5]])
    return (np.array(expc6 / theta), theta)

def MatrixExp6(se3mat, formula=2):
    """Computes the matrix exponential of an se3 representation of exponential coordinates e^ξ_hat*θ

    :param se3mat: A matrix in se3
    :param fomula: (Optional) Choose formula 1 or 2. Default=1

        Eq1. e^ξθ = [e^w_hat*θ (I-e^hat*θ)(w_hat*v)+w*w^T*v*θ; 0 1]
        
        Eq2. e^ξθ = [e^w_hat*θ (Iθ + (1-cosθ)w_hat + (θ-sinθ)w_hat^2)v; 0 1], w is unit vector.
    :return: The matrix exponential of se3mat

    Example Input:
        se3mat = np.array([[0,          0,           0,          0],
                           [0,          0, -1.57079632, 2.35619449],
                           [0, 1.57079632,           0, 2.35619449],
                           [0,          0,           0,          0]])
    Output:
        np.array([[1.0, 0.0,  0.0, 0.0],
                  [0.0, 0.0, -1.0, 0.0],
                  [0.0, 1.0,  0.0, 3.0],
                  [  0,   0,    0,   1]])
    """
    se3mat = np.array(se3mat)
    omgtheta = so3ToVec(se3mat[0: 3, 0: 3])
    if NearZero(np.linalg.norm(omgtheta)):
        return np.r_[np.c_[np.eye(3), se3mat[0: 3, 3]], [[0, 0, 0, 1]]]
    else:
        theta = AxisAng3(omgtheta)[1]
        rotmat = MatrixExp3(se3mat[0: 3, 0: 3]) # rotation matrix
        if formula == 1:
            omgmat = se3mat[0: 3, 0: 3] / theta
            omg = omgtheta / theta
            return np.r_[np.c_[rotmat, np.dot((np.eye(3) - rotmat), np.cross(omg,se3mat[0: 3, 3]))/theta + np.dot(omg,se3mat[0: 3, 3]) * omgtheta],
                        [[0, 0, 0, 1]]]
        elif formula == 2:
            omgmat = se3mat[0: 3, 0: 3] / theta
            return np.r_[np.c_[rotmat,
                            np.dot(np.eye(3)*theta + (1-np.cos(theta))*omgmat + (theta-np.sin(theta))*np.dot(omgmat,omgmat),
                                    se3mat[0: 3, 3]) / theta],
                        [[0, 0, 0, 1]]]

def MatrixLog6(T):
    """Computes the matrix logarithm of a homogeneous transformation matrix

    :param R: A matrix in SE3 = [R p; 0 1]
    :return: The matrix logarithm of T = [w_hat*θ v; 0 0]

    Example Input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        np.array([[0,          0,           0,           0]
                  [0,          0, -1.57079633,  2.35619449]
                  [0, 1.57079633,           0,  2.35619449]
                  [0,          0,           0,           0]])
    """
    R, p = TransToRp(T)
    omgmat = MatrixLog3(R)
    if np.array_equal(omgmat, np.zeros((3, 3))):
        return np.r_[np.c_[np.zeros((3, 3)),
                           [T[0][3], T[1][3], T[2][3]]],
                     [[0, 0, 0, 0]]]
    else:
        theta = np.arccos((np.trace(R) - 1) / 2.0)
        return np.r_[np.c_[omgmat,
                           np.dot(np.eye(3) - omgmat / 2.0 \
                           + (1.0 / theta - 1.0 / np.tan(theta / 2.0) / 2) \
                              * np.dot(omgmat,omgmat) / theta,[T[0][3],
                                                               T[1][3],
                                                               T[2][3]])],
                     [[0, 0, 0, 0]]]
    
def FKinBody(M, Blist, thetalist):
    """Computes forward kinematics in the body frame for an open chain robot

    :param M: The home configuration (position and orientation) of the end-effector
    :param Blist: The joint screw axes with notation [v w] in the end-effector frame when the manipulator is at the home position, in the format of a matrix with axes as the columns
    :param thetalist: A list of joint coordinates
    :return: A homogeneous transformation matrix representing the end-effector frame when the joints are at the specified coordinates (i.t.o Body Frame)

    Example Input:
        M = np.array([[-1, 0,  0, 0],
                      [ 0, 1,  0, 6],
                      [ 0, 0, -1, 2],
                      [ 0, 0,  0, 1]])
        Blist = np.array([[2, 0,   0, 0, 0, -1],
                          [0, 1,   0, 0, 0,  0],
                          [0, 0, 0.1, 0, 0,  1]]).T
        thetalist = np.array([np.pi / 2.0, 3, np.pi])
    Output:
        np.array([[0, 1,  0,         -5],
                  [1, 0,  0,          4],
                  [0, 0, -1, 1.68584073],
                  [0, 0,  0,          1]])
    """
    T = np.array(M)
    for i in range(len(thetalist)):
        T = np.dot(T, MatrixExp6(VecTose3(np.array(Blist)[:, i] \
                                          * thetalist[i])))
    return T

def FKinSpace(M, Slist, thetalist):
    """Computes forward kinematics in the space frame for an open chain robot

    :param M: The home configuration (position and orientation) of the end-effector
    :param Slist: The joint screw axes in the space frame when the manipulator is at the home position, in the format of a matrix with axes as the columns
    :param thetalist: A list of joint coordinates
    :return: A homogeneous transformation matrix representing the end-effector frame when the joints are at the specified coordinates (i.t.o Space Frame)

    Example Input:
        M = np.array([[-1, 0,  0, 0],
                      [ 0, 1,  0, 6],
                      [ 0, 0, -1, 2],
                      [ 0, 0,  0, 1]])
        Slist = np.array([[ 4, 0,    0, 0, 0,  1],
                          [ 0, 1,    0, 0, 0,  0],
                          [-6, 0, -0.1, 0, 0, -1]]).T
        thetalist = np.array([np.pi / 2.0, 3, np.pi])
    Output:
        np.array([[0, 1,  0,         -5],
                  [1, 0,  0,          4],
                  [0, 0, -1, 1.68584073],
                  [0, 0,  0,          1]])
    """
    T = np.array(M)
    for i in range(len(thetalist) - 1, -1, -1):
        T = np.dot(MatrixExp6(VecTose3(np.array(Slist)[:, i] \
                                       * thetalist[i])), T)
    return T

def JacobianBody(Blist, thetalist):
    """Computes the body Jacobian for an open chain robot

    :param Blist: The joint screw axes in the end-effector frame when the manipulator is at the home position, in the format of a matrix with axes as the columns
    :param thetalist: A list of joint coordinates
    :return: The body Jacobian corresponding to the inputs (6xn real numbers)

    Example Input:
        Blist = np.array([[   0,   0.2, 0.2, 0, 0, 1],
                          [   2,     0,   3, 1, 0, 0],
                          [   0,     2,   1, 0, 1, 0],
                          [ 0.2,   0.3, 0.4, 1, 0, 0]]).T
        thetalist = np.array([0.2, 1.1, 0.1, 1.2])
    Output:
        np.array([[ 2.32586047,    1.66809,  0.56410831, 0.2]
                  [-1.44321167, 2.94561275,  1.43306521, 0.3]
                  [-2.06639565, 1.82881722, -1.58868628, 0.4]
                  [-0.04528405, 0.99500417,           0,   1]
                  [ 0.74359313, 0.09304865,  0.36235775,   0]
                  [-0.66709716, 0.03617541, -0.93203909,   0]])
    """
    Jb = np.array(Blist).copy().astype(float)
    T = np.eye(4)
    for i in range(len(thetalist) - 2, -1, -1):
        T = np.dot(T,MatrixExp6(VecTose3(np.array(Blist)[:, i + 1] \
                                         * -thetalist[i + 1])))
        Jb[:, i] = np.dot(Adjoint(T), np.array(Blist)[:, i])
    return Jb

def JacobianSpace(Slist, thetalist):
    """Computes the space Jacobian for an open chain robot

    :param Slist: The joint screw axes in the space frame when the manipulator is at the home position, in the format of a matrix with axes as the columns
    :param thetalist: A list of joint coordinates
    :return: The space Jacobian corresponding to the inputs (6xn real numbers)

    Example Input:
        Slist = np.array([[  0, 0.2, 0.2, 0, 0, 1],
                          [  2,   0,   3, 1, 0, 0],
                          [  0,   2,   1, 0, 1, 0],
                          [0.2, 0.3, 0.4, 1, 0, 0]]).T
        thetalist = np.array([0.2, 1.1, 0.1, 1.2])
    Output:
        np.array([[  0, 1.95218638, -2.21635216, -0.51161537]
                  [0.2, 0.43654132, -2.43712573,  2.77535713]
                  [0.2, 2.96026613,  3.23573065,  2.22512443]
                  [  0, 0.98006658, -0.09011564,  0.95749426]
                  [  0, 0.19866933,   0.4445544,  0.28487557]
                  [  1,          0,  0.89120736, -0.04528405]])
    """
    Js = np.array(Slist).copy().astype(float)
    T = np.eye(4)
    for i in range(1, len(thetalist)):
        T = np.dot(T, MatrixExp6(VecTose3(np.array(Slist)[:, i - 1] \
                                * thetalist[i - 1])))
        Js[:, i] = np.dot(Adjoint(T), np.array(Slist)[:, i])
    return Js

def IKinBody(Blist, M, T, thetalist0, eomg, ev, maxiter=1000,verbose=False):
    """Computes inverse kinematics in the body frame for an open chain robot

    :param Blist: The joint screw axes in the end-effector frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param M: The home configuration of the end-effector
    :param T: The desired end-effector configuration Tsd
    :param thetalist0: An initial guess of joint angles that are close to
                       satisfying Tsd
    :param eomg: A small positive tolerance on the end-effector orientation
                 error. The returned joint angles must give an end-effector
                 orientation error less than eomg
    :param ev: A small positive tolerance on the end-effector linear position
               error. The returned joint angles must give an end-effector
               position error less than ev
    :param maxiter: (Optional) Maximum number of iterations. Default=1000.
    :return thetalist: Joint angles that achieve T within the specified
                       tolerances,
    :return success: A logical value where TRUE means that the function found
                     a solution and FALSE means that it ran through the set
                     number of maximum iterations without finding a solution
                     within the tolerances eomg and ev.
    Uses an iterative Newton-Raphson root-finding method.
    The maximum number of iterations before the algorithm is terminated has
    been hardcoded in as a variable called maxiterations. It is set to 20 at
    the start of the function, but can be changed if needed.

    Example Input:
        Blist = np.array([[2, 0,   0, 0, 0, -1],
                          [0, 1,   0, 0, 0,  0],
                          [0, 0, 0.1, 0, 0,  1]]).T
        M = np.array([[-1, 0,  0, 0],
                      [ 0, 1,  0, 6],
                      [ 0, 0, -1, 2],
                      [ 0, 0,  0, 1]])
        T = np.array([[0, 1,  0,     -5],
                      [1, 0,  0,      4],
                      [0, 0, -1, 1.6858],
                      [0, 0,  0,      1]])
        thetalist0 = np.array([1.5, 2.5, 3])
        eomg = 0.01
        ev = 0.001
    Output:
        (np.array([1.57073819, 2.999667, 3.14153913]), True)
    """
    thetalist = np.array(thetalist0).copy()
    i = 0
    maxiterations = maxiter
    Vb = se3ToVec(MatrixLog6(np.dot(TransInv(FKinBody(M, Blist, \
                                                      thetalist)), T)))
    err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > ev \
          or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > eomg
    
    if verbose:
        with tqdm(total=maxiterations, desc="[IKinBody] Root Finding with iterative Newton-Raphson") as pbar:
            while err and i < maxiterations:
                pbar.update(1)
                thetalist = thetalist \
                            + np.dot(np.linalg.pinv(JacobianBody(Blist, \
                                                                thetalist)), Vb)
                i = i + 1
                Vb \
                = se3ToVec(MatrixLog6(np.dot(TransInv(FKinBody(M, Blist, \
                                                            thetalist)), T)))
                err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > ev \
                    or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > eomg
    else:
        while err and i < maxiterations:
            thetalist = thetalist \
                        + np.dot(np.linalg.pinv(JacobianBody(Blist, \
                                                            thetalist)), Vb)
            i = i + 1
            Vb \
            = se3ToVec(MatrixLog6(np.dot(TransInv(FKinBody(M, Blist, \
                                                        thetalist)), T)))
            err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > ev \
                or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > eomg
    return (thetalist, not err)

def IKinSpace(Slist, M, T, thetalist0, eomg, ev, maxiter=1000):
    """Computes inverse kinematics in the space frame for an open chain robot

    :param Slist: The joint screw axes in the space frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param M: The home configuration of the end-effector
    :param T: The desired end-effector configuration Tsd
    :param thetalist0: An initial guess of joint angles that are close to
                       satisfying Tsd
    :param eomg: A small positive tolerance on the end-effector orientation
                 error. The returned joint angles must give an end-effector
                 orientation error less than eomg
    :param ev: A small positive tolerance on the end-effector linear position
               error. The returned joint angles must give an end-effector
               position error less than ev
    :param maxiter: (Optional) Maximum number of iterations. Default=1000.
    :return thetalist: Joint angles that achieve T within the specified
                       tolerances,
    :return success: A logical value where TRUE means that the function found
                     a solution and FALSE means that it ran through the set
                     number of maximum iterations without finding a solution
                     within the tolerances eomg and ev.
    Uses an iterative Newton-Raphson root-finding method.
    The maximum number of iterations before the algorithm is terminated has
    been hardcoded in as a variable called maxiterations. It is set to 20 at
    the start of the function, but can be changed if needed.

    Example Input:
        Slist = np.array([[ 4, 0,    0, 0, 0,  1],
                          [ 0, 1,    0, 0, 0,  0],
                          [-6, 0, -0.1, 0, 0, -1]]).T
        M = np.array([[-1, 0,  0, 0],
                      [ 0, 1,  0, 6],
                      [ 0, 0, -1, 2],
                      [ 0, 0,  0, 1]])
        T = np.array([[0, 1,  0,     -5],
                      [1, 0,  0,      4],
                      [0, 0, -1, 1.6858],
                      [0, 0,  0,      1]])
        thetalist0 = np.array([1.5, 2.5, 3])
        eomg = 0.01
        ev = 0.001
    Output:
        (np.array([ 1.57073783,  2.99966384,  3.1415342 ]), True)
    """
    thetalist = np.array(thetalist0).copy()
    i = 0
    maxiterations = maxiter
    Tsb = FKinSpace(M,Slist, thetalist)
    Vs = np.dot(Adjoint(Tsb), \
                se3ToVec(MatrixLog6(np.dot(TransInv(Tsb), T))))
    err = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > ev \
          or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > eomg
    
    with tqdm(total=maxiterations, desc="[IKinSpace] Root Finding with iterative Newton-Raphson") as pbar:
        while err and i < maxiterations:
            pbar.update(1)
            thetalist = thetalist \
                        + np.dot(np.linalg.pinv(JacobianSpace(Slist, \
                                                            thetalist)), Vs)
            i = i + 1
            Tsb = FKinSpace(M, Slist, thetalist)
            Vs = np.dot(Adjoint(Tsb), \
                        se3ToVec(MatrixLog6(np.dot(TransInv(Tsb), T))))
            err = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > ev \
                or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > eomg
    return (thetalist, not err)

def CubicTimeScaling(Tf, t):
    """Computes s(t) for a cubic time scaling

    :param Tf: Total time of the motion in seconds from rest to rest
    :param t: The current time t satisfying 0 < t < Tf
    :return: The path parameter s(t) corresponding to a third-order
             polynomial motion that begins and ends at zero velocity

    Example Input:
        Tf = 2
        t = 0.6
    Output:
        0.216
    """
    return 3 * (1.0 * t / Tf) ** 2 - 2 * (1.0 * t / Tf) ** 3

def QuinticTimeScaling(Tf, t):
    """Computes s(t) for a quintic time scaling

    :param Tf: Total time of the motion in seconds from rest to rest
    :param t: The current time t satisfying 0 < t < Tf
    :return: The path parameter s(t) corresponding to a fifth-order
             polynomial motion that begins and ends at zero velocity and zero
             acceleration

    Example Input:
        Tf = 2
        t = 0.6
    Output:
        0.16308
    """
    return 10 * (1.0 * t / Tf) ** 3 - 15 * (1.0 * t / Tf) ** 4 \
           + 6 * (1.0 * t / Tf) ** 5

def JointTrajectory(thetastart, thetaend, Tf, N, method):
    """Computes a straight-line trajectory in joint space

    :param thetastart: The initial joint variables
    :param thetaend: The final joint variables
    :param Tf: Total time of the motion in seconds from rest to rest
    :param N: The number of points N > 1 (Start and stop) in the discrete
              representation of the trajectory
    :param method: The time-scaling method, where 3 indicates cubic (third-
                   order polynomial) time scaling and 5 indicates quintic
                   (fifth-order polynomial) time scaling
    :return: A trajectory as an N x n matrix, where each row is an n-vector
             of joint variables at an instant in time. The first row is
             thetastart and the Nth row is thetaend . The elapsed time
             between each row is Tf / (N - 1)

    Example Input:
        thetastart = np.array([1, 0, 0, 1, 1, 0.2, 0,1])
        thetaend = np.array([1.2, 0.5, 0.6, 1.1, 2, 2, 0.9, 1])
        Tf = 4
        N = 6
        method = 3
    Output:
        np.array([[     1,     0,      0,      1,     1,    0.2,      0, 1]
                  [1.0208, 0.052, 0.0624, 1.0104, 1.104, 0.3872, 0.0936, 1]
                  [1.0704, 0.176, 0.2112, 1.0352, 1.352, 0.8336, 0.3168, 1]
                  [1.1296, 0.324, 0.3888, 1.0648, 1.648, 1.3664, 0.5832, 1]
                  [1.1792, 0.448, 0.5376, 1.0896, 1.896, 1.8128, 0.8064, 1]
                  [   1.2,   0.5,    0.6,    1.1,     2,      2,    0.9, 1]])
    """
    N = int(N)
    timegap = Tf / (N - 1.0)
    traj = np.zeros((len(thetastart), N))
    for i in range(N):
        if method == 3:
            s = CubicTimeScaling(Tf, timegap * i)
        else:
            s = QuinticTimeScaling(Tf, timegap * i)
        traj[:, i] = s * np.array(thetaend) + (1 - s) * np.array(thetastart)
    traj = np.array(traj).T
    return traj

def CartesianTrajectory(Xstart, Xend, Tf, N, method):
    """Computes a trajectory as a list of N SE(3) matrices corresponding to
    the origin of the end-effector frame following a straight line

    :param Xstart: The initial end-effector configuration
    :param Xend: The final end-effector configuration
    :param Tf: Total time of the motion in seconds from rest to rest
    :param N: The number of points N > 1 (Start and stop) in the discrete
              representation of the trajectory
    :param method: The time-scaling method, where 3 indicates cubic (third-
                   order polynomial) time scaling and 5 indicates quintic
                   (fifth-order polynomial) time scaling
    :return: The discretized trajectory as a list of N matrices in SE(3)
             separated in time by Tf/(N-1). The first in the list is Xstart
             and the Nth is Xend
    This function is similar to ScrewTrajectory, except the origin of the
    end-effector frame follows a straight line, decoupled from the rotational
    motion.

    Example Input:
        Xstart = np.array([[1, 0, 0, 1],
                           [0, 1, 0, 0],
                           [0, 0, 1, 1],
                           [0, 0, 0, 1]])
        Xend = np.array([[0, 0, 1, 0.1],
                         [1, 0, 0,   0],
                         [0, 1, 0, 4.1],
                         [0, 0, 0,   1]])
        Tf = 5
        N = 4
        method = 5
    Output:
        [np.array([[1, 0, 0, 1]
                   [0, 1, 0, 0]
                   [0, 0, 1, 1]
                   [0, 0, 0, 1]]),
         np.array([[ 0.937, -0.214,  0.277, 0.811]
                   [ 0.277,  0.937, -0.214,     0]
                   [-0.214,  0.277,  0.937, 1.651]
                   [     0,      0,      0,     1]]),
         np.array([[ 0.277, -0.214,  0.937, 0.289]
                   [ 0.937,  0.277, -0.214,     0]
                   [-0.214,  0.937,  0.277, 3.449]
                   [     0,      0,      0,     1]]),
         np.array([[0, 0, 1, 0.1]
                   [1, 0, 0,   0]
                   [0, 1, 0, 4.1]
                   [0, 0, 0,   1]])]
    """
    N = int(N)
    timegap = Tf / (N - 1.0)
    traj = [[None]] * N
    Rstart, pstart = TransToRp(Xstart)
    Rend, pend = TransToRp(Xend)
    for i in range(N):
        if method == 3:
            s = CubicTimeScaling(Tf, timegap * i)
        else:
            s = QuinticTimeScaling(Tf, timegap * i)
        traj[i] \
        = np.r_[np.c_[np.dot(Rstart, \
        MatrixExp3(MatrixLog3(np.dot(np.array(Rstart).T,Rend)) * s)), \
                   s * np.array(pend) + (1 - s) * np.array(pstart)], \
                   [[0, 0, 0, 1]]]
    return traj