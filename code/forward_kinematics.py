# Python 2/3 compatibility
from __future__ import print_function

import os
import argparse

import numpy as np
from urdf_parser_py.urdf import URDF
from tf.transformations import euler_matrix, rotation_matrix, translation_matrix, decompose_matrix

from utils import block_printing

PKG_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


@block_printing
def parse_urdf(filename):
    return URDF.from_xml_file(os.path.join(PKG_PATH, filename))


def find_joint(robot, name):
    for joint in robot.joints:
        if joint.name == name:
            return joint
    return None


def get_joint_data(robot, name):
    joint = find_joint(robot, name)
    T = translation_matrix(joint.origin.xyz)
    r, p, y = joint.origin.rpy
    R = euler_matrix(r, p, y)
    axis = np.asarray(joint.axis)
    return np.matmul(T, R), axis


def getWristPose(joint_angle_list, robot, joint_names, intermediate_pose=False):
    '''Get the wrist pose for given joint angle configuration.

    joint_angle_list: List of joint angles to get final wrist pose for
    kwargs: Other keyword arguments use as required.

    TODO: You can change the signature of this method to pass in other objects,
    such as the path to the URDF file or a configuration of your URDF file that
    has been read previously into memory. 

    Return: List of 16 values which represent the joint wrist pose 
    obtained from the End-Effector Transformation matrix using column-major
    ordering.
    '''

    pose = np.eye(4)
    intermediate_poses = []
    for joint, angle in zip(joint_names, joint_angle_list):
        H_joint, axis = get_joint_data(robot, joint)
        rot = rotation_matrix(angle, axis)
        pose = np.matmul(pose, np.matmul(H_joint, rot))
        intermediate_poses.append((pose, axis))
    
    if intermediate_pose:
        return intermediate_poses
    else:
        return pose


def getWristJacobian(joint_angle_list, robot, joint_names):
    '''Get the wrist jacobian for given joint angle configuration.

    joint_angle_list: List of joint angles to get final wrist pose for
    kwargs: Other keyword arguments use as required.

    TODO: You can change the signature of this method to pass in other objects,
    such as the wrist pose for this configuration or path to the URDF
    file or a configuration of your URDF file that has been read previously
    into memory. 

    Return: List of 16 values which represent the joint wrist pose 
    obtained from the End-Effector Transformation matrix using column-major
    ordering.
    '''
    jacobian = np.zeros((6, 5))
    joint_angles = np.asarray(joint_angle_list)
    
    poses = getWristPose(joint_angle_list, robot, joint_names, intermediate_pose=True)
    for i, (pose, axis) in enumerate(poses):
        jacobian[3:6, i] = np.matmul(pose[:3, :3], axis)
        jacobian[0:3, i] = np.cross(jacobian[3:6, i].T, poses[-1][0][:3, -1] - pose[:3, -1])

    return jacobian
    

def main(args):
    # Joint angles
    joint_angles = args.joints
    assert len(joint_angles) == 5, 'Incorrect number of joints specified.'

    # Joint names
    joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5']

    # Parse URDF XML
    robot = parse_urdf('urdf/locobot_description_v3.urdf')

    # Compute end effector pose and jacobian
    pose = getWristPose(joint_angles, robot, joint_names)
    jacobian = getWristJacobian(joint_angles, robot, joint_names)

    print('Wrist Pose:\n{}'.format(np.array_str(np.array(pose), precision=3)))
    print('Jacobian:\n{}'.format(np.array_str(np.array(jacobian), precision=3)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Get wrist pose using forward kinematics')
    parser.add_argument('--joints', type=float, nargs='+',
        required=True, help='Joint angles to get wrist position for.')
    args = parser.parse_args()
    
    main(args)
