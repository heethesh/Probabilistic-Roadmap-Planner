'''
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
import sys
import time
import traceback

import numpy as np
from tf.transformations import euler_matrix, rotation_matrix, translation_matrix

import utils
import forward_kinematics as fk
import locobot_joint_ctrl as bot
from utilities import vrep_utils as vu
from cuboid_collision import Cuboid, CollisionChecker

# Handle names
LINK_NAMES = ['arm_base_link_joint', 'shoulder_link', 'elbow_link', 'forearm_link', 'wrist_link', 'gripper_link', 'finger_r', 'finger_l']
JOINT_NAMES = ['arm_base_link_joint', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']
OBJECT_NAMES = ['cuboid_0', 'cuboid_1', 'cuboid_2', 'cuboid_3', 'cuboid_4', 'cuboid_5']

# Collision checker
collision_checker = CollisionChecker()

def planner(clientID, robot):
    # Get collision cuboids
    link_cuboids = utils.get_cuboids(clientID, LINK_NAMES)
    object_cuboids = utils.get_cuboids(clientID, OBJECT_NAMES, append=False)

    # Display collision cuboids
    collision_checker.display_cuboids(link_cuboids + object_cuboids)


def main():
    # Connect to V-REP
    print ('Connecting to V-REP...')
    clientID = vu.connect_to_vrep()
    print ('Connected.')

    # Reset simulation in case something was running
    vu.reset_sim(clientID)
    
    # Initial control inputs are zero
    vu.set_arm_joint_target_velocities(clientID, np.zeros(vu.N_ARM_JOINTS))

    # Despite the name, this sets the maximum allowable joint force
    vu.set_arm_joint_forces(clientID, 50.*np.ones(vu.N_ARM_JOINTS))

    # One step to process the above settings
    vu.step_sim(clientID, 1)

    # Parse URDF XML
    robot = fk.parse_urdf('urdf/locobot_description_v3.urdf')

    # Joint targets, radians for revolute joints and meters for prismatic joints
    gripper_targets = np.asarray([[-0.03, 0.03], [-0.03, 0.03]])
    joint_targets = np.radians([[-80, 0, 0, 0, 0], [0, 60, -75, -75, 0]])
    
    try:
        planner(clientID, None)
    except Exception as e:
        print(e)
        traceback.print_exc()
    finally:
        vu.stop_sim(clientID)


if __name__ == '__main__':
    main()
