'''
16-662 Robot Autonomy (Spring 2019)
Homework 2 - Motion Planning and Collision Avoidance
Author: Heethesh Vhavle
Email: heethesh@cmu.edu
Version: 1.0.0

The order of the targets are as follows:
    joint_1 / revolute  / arm_base_link <- shoulder_link
    joint_2 / revolute  / shoulder_link <- elbow_link
    joint_3 / revolute  / elbow_link    <- forearm_link
    joint_4 / revolute  / forearm_link  <- wrist_link
    joint_5 / revolute  / wrist_link    <- gripper_link
    joint_6 / prismatic / gripper_link  <- finger_r
    joint_7 / prismatic / gripper_link  <- finger_l
'''

# Python 2/3 compatibility
from __future__ import print_function

import os
import argparse

import numpy as np
from tf.transformations import euler_matrix, rotation_matrix, translation_matrix, decompose_matrix

from cuboid_collision import Cuboid
from utils import URDFRobot, print_pose, JOINT_NAMES


class Joint(object):
    def __init__(self, origin, angles, axis, name=None, vrep=False):
        # Check dimensions
        assert len(axis) == 3
        assert len(origin) == 3
        assert len(angles) == 3 or len(angles) == 4

        # Joint name
        if name: self.name = name

        # Origin
        self.origin = np.asarray(origin)
        self.origin_matrix = translation_matrix(origin)

        # Orientation
        self.angles = np.asarray(angles)
        if vrep:
            self.rot_matrix = Cuboid.vrep_rotation_matrix(self.angles)
        else:
            if len(angles) == 3: self.rot_matrix = euler_matrix(angles[0], angles[1], angles[2])
            elif len(angles) == 4: self.rot_matrix = quaternion_matrix(angles)

        # Joint axis
        self.axis = axis


class PrismaticJoint(Joint):
    def __init__(self, origin, angles, axis, offset=0, name=None):
        # Init other parameters
        super(PrismaticJoint, self).__init__(origin, angles, axis, name)
        self.set_joint_offset(offset)

    def set_joint_offset(self, offset):
        # Prismatic joint properties
        self.offset = offset
        self.offset_xyz = np.full_like(self.axis, self.offset) * self.axis
        self.offset_matrix = translation_matrix(self.offset_xyz)


class RevoluteJoint(Joint):
    def __init__(self, origin, angles, axis, offset=0, name=None):
        # Init other parameters
        super(RevoluteJoint, self).__init__(origin, angles, axis, name)
        self.set_joint_offset(offset)

    def set_joint_offset(self, offset):
        # Revolute joint properties
        self.offset = offset
        self.offset_matrix = rotation_matrix(self.offset, self.axis)


class ForwardKinematicsSolver:
    def __init__(self, robot):
        # URDF object
        self.robot = robot
        
        # Joint objects
        self.joints = []

        # Link to joint transforms
        self.link_to_joints = []

    def update_joints(self, joint_angles, joint_poses=None):
        # print('>>> Joint Angles:', np.degrees(joint_angles))
        # Check dimensions

        if self.joints:
            for joint, offset in zip(self.joints, joint_angles):
                joint.set_joint_offset(offset)
        else:
            assert len(joint_angles) == len(joint_poses)
            for i, joint_name in enumerate(JOINT_NAMES):
                # Get joint axis from URDF
                axis = self.robot.get_joint_axis(joint_name)

                # Select joint type 
                if i < 5: JointType = RevoluteJoint
                else: JointType = PrismaticJoint
                
                # Create joint object
                self.joints.append(JointType(origin=joint_poses[i][0], angles=joint_poses[i][1],
                    axis=axis, offset=joint_angles[i], name=joint_name))

    def compute(self, links, joint_angles, joint_poses=None, setup=False):
        # Check dimensions
        assert len(joint_angles) >= 4

        # Update joint angles
        self.update_joints(joint_angles, joint_poses)
        
        # Links and joint poses
        pose = np.eye(4)
        link_cuboids = links[:1]

        # Compute forward kinematics
        for i, joint in enumerate(self.joints):
            # Compute joint pose
            R, T, axis = self.robot.get_joint_pose(joint.name)
            pose = np.matmul(pose, np.matmul(np.matmul(R, T), joint.offset_matrix))
            
            # Update arm joints
            if i < 5:
                wrist_pose = pose

            # Setup link to joint transforms
            if setup:
                self.link_to_joints.append(np.matmul(np.linalg.inv(wrist_pose), links[i + 1].transform_matrix))

            # Compute link pose
            else:
                link_pose = np.matmul(wrist_pose, self.link_to_joints[i])
                _, _, R, T, _ = decompose_matrix(link_pose)
                
                # Create link cuboids
                link_cuboids.append(Cuboid(T, R, links[i + 1].dxyz, links[i + 1].name))

        # Return wrist pose and link cuboids
        _, _, R, T, _ = decompose_matrix(wrist_pose)
        return R, T, link_cuboids

    # Backward compatible
    @staticmethod
    def getWristPose(joint_angle_list, robot, joint_names, intermediate_pose=False):
        '''
        Get the wrist pose for given joint angle configuration.

        joint_angle_list: List of joint angles to get final wrist pose for
        kwargs: Other keyword arguments use as required.

        Return: List of 16 values which represent the joint wrist pose 
        obtained from the End-Effector Transformation matrix using column-major
        ordering.
        '''

        pose = np.eye(4)
        intermediate_poses = []

        for joint, angle in zip(joint_names, joint_angle_list):
            R, T, axis = robot.get_joint_pose(joint)
            A = rotation_matrix(angle, axis)
            pose = np.matmul(np.matmul(pose, np.matmul(R, T)), A)
            intermediate_poses.append((pose, axis))
        
        if intermediate_pose:
            return intermediate_poses
        else:
            return pose

    # Backward compatible
    @staticmethod
    def getWristJacobian(joint_angle_list, robot, joint_names):
        '''
        Get the wrist Jacobian for given joint angle configuration.

        joint_angle_list: List of joint angles to get final wrist pose for
        kwargs: Other keyword arguments use as required.

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
    joint_names = JOINT_NAMES[:-2]

    # Parse URDF XML
    robot = URDFRobot('urdf/locobot_description_v3.urdf')

    # Compute end effector pose and Jacobian
    pose = ForwardKinematicsSolver.getWristPose(joint_angles, robot, joint_names)
    jacobian = ForwardKinematicsSolver.getWristJacobian(joint_angles, robot, joint_names)

    print('Wrist Pose:\n{}'.format(np.array_str(np.array(pose), precision=3)))
    print('Jacobian:\n{}'.format(np.array_str(np.array(jacobian), precision=3)))


def test():
    # Joint info
    joint_names = JOINT_NAMES[:-2]
    joint_angles = np.radians([-80, 0, 0, 0, 0])

    # Parse URDF XML
    robot = URDFRobot('urdf/locobot_description_v3.urdf')

    # Compute end effector pose and Jacobian
    poses = ForwardKinematicsSolver.getWristPose(joint_angles, robot, joint_names, intermediate_pose=True)
    for i, (pose, axis) in enumerate(poses): print_pose(pose)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(
    #     description='Get wrist pose using forward kinematics')
    # parser.add_argument('--joints', type=float, nargs='+',
    #     required=True, help='Joint angles to get wrist position for.')
    # args = parser.parse_args()
    
    # main(args)
    test()
