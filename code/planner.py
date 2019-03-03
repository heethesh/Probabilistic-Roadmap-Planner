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

import numpy as np
from tf.transformations import euler_matrix, rotation_matrix, translation_matrix

import forward_kinematics
import locobot_joint_ctrl
from utilities import vrep_utils as vu


JOINT_NAMES = ['arm_base_link_joint', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']


def get_bounding_boxes():
    pass


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
    vu.step_sim(clientID)
    
    get_bounding_boxes()

    vu.stop_sim(clientID)

if __name__ == '__main__':
    main()
