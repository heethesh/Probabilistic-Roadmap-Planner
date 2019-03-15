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

import utils
import locobot_joint_ctrl
from utilities import vrep_utils as vu
from cuboid_collision import CollisionChecker
from forward_kinematics import ForwardKinematicsSolver


class ProbabilisticRoadMap:
    def __init__(self, clientID, joint_targets, gripper_targets, urdf):
        # VREP client ID
        self.clientID = clientID

        # Targets
        self.joint_targets = joint_targets
        self.gripper_targets = gripper_targets

        # SAT collision checker
        self.collision_checker = CollisionChecker()

        # Get collision cuboids
        self.link_cuboids = utils.get_cuboids(self.clientID, utils.LINK_NAMES)
        self.obstacle_cuboids = utils.get_cuboids(self.clientID, utils.OBSTACLE_NAMES, append=False)
        
        # URDF parser
        self.robot = utils.URDFRobot(urdf)

        # Forward kinematics solver
        self.fksolver = ForwardKinematicsSolver(self.robot)

        # Initialize
        self.setup()
    
    def setup(self):
        # Get initial joint positions and link cuboids from VREP
        joint_angles = vu.get_arm_joint_positions(self.clientID)
        joint_poses = utils.get_joint_pose_vrep(self.clientID)
        self.fksolver.compute(self.link_cuboids, joint_angles, joint_poses, setup=True)

        # Simulation no longer required for planning
        vu.stop_sim(self.clientID)

    def plan(self):
        
        # Get joint objects
        joint_angles = np.radians([0, 45, -90, 10, 10])

        # Get poses after forward kinematics
        self.link_cuboids = self.fksolver.compute(self.link_cuboids, joint_angles)

    def get_random_sample(self):
        # TODO
        limits = [ self.robot.get_joint_limits(j) for j in utils.JOINT_NAMES ]
        
    def collision_free(self):
        for link in link_cuboids:
            for obstacle in obstacle_cuboids:
                if not self.collision_checker.detect_collision_optimized(link, obstacle):
                    return False
        return True

    def local_planner(self, start, end, n_points=5):
        # Check dimensions
        assert len(start) == len(end)

        # Interpolate to form a local path
        local_path = np.linspace(start, end, num=n_points)
        
        for point in local_path:
            # Get poses after forward kinematics
            self.link_cuboids = self.fksolver.compute(self.link_cuboids, point)
            
            # Check for collisions along the path
            if not self.collision_free():
                return False

        # Path planning successful
        return True

    def path_planner(self):
        pass

    def distance(self, a, b):
        pass

    def display_cuboids(self, obstacles=False):
        # Display collision cuboids
        if obstacles:
            self.collision_checker.display_cuboids(self.link_cuboids + self.obstacle_cuboids)
        else:
            self.collision_checker.display_cuboids(self.link_cuboids)


def main():
    # Connect to V-REP
    print ('Connecting to V-REP...')
    clientID = vu.connect_to_vrep()
    print ('Connected.')

    # Reset simulation in case something was running
    vu.reset_sim(clientID)
    
    # Initial control inputs are zero
    # vu.set_arm_joint_positions(clientID, np.zeros(vu.N_ARM_JOINTS))
    # vu.set_arm_joint_positions(clientID, np.radians([-80, 10, 10, 10, 10, 0, 0]))
    vu.set_arm_joint_target_velocities(clientID, np.zeros(vu.N_ARM_JOINTS))

    # Despite the name, this sets the maximum allowable joint force
    # vu.set_arm_joint_forces(clientID, 50.*np.ones(vu.N_ARM_JOINTS))

    # One step to process the above settings
    # vu.step_sim(clientID, 1)

    # Joint targets, radians for revolute joints and meters for prismatic joints
    gripper_targets = np.asarray([[-0.03, 0.03], [-0.03, 0.03]])
    joint_targets = np.radians([[-80, 0, 0, 0, 0], [0, 60, -75, -75, 0]])
    
    try:
        prm_planner = ProbabilisticRoadMap(clientID, joint_targets, gripper_targets,
            urdf='urdf/locobot_description_v3.urdf')    
        # prm_planner.plan()
        prm_planner.local_planner([0, 1, 2], [2, 3, 4])
    except Exception as e:
        print(e)
        traceback.print_exc()
    finally:
        vu.stop_sim(clientID)


if __name__ == '__main__':
    main()
