'''
16-662 Robot Autonomy (Spring 2019)
Homework 2 - Motion Planning and Collision Avoidance
Author: Heethesh Vhavle
Email: heethesh@cmu.edu
Version: 1.0.0
'''

# Python 2/3 compatibility
from __future__ import print_function

# Import system libraries
import os
import sys
import time
import argparse
import traceback

# Modify the following lines if you have problems importing the V-REP utilities
cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(os.path.join(cwd, 'lib'))
sys.path.append(os.path.join(cwd, 'utilities'))

# Import application libraries
import numpy as np
import vrep_utils as vu
import matplotlib.pyplot as plt

###############################################################################

class ArmController:
    def __init__(self, clientID):
        # Do not modify the following variables
        self.history = {'timestamp': [],
                        'joint_feedback': [],
                        'joint_target': [],
                        'ctrl_commands': []}
        self._target_joint_positions = None
        
        # Defined variables
        self.clientID = clientID
        self.first_loop = True
        self.wait_time = 2
        self.converged_time = 0
        self.converged_wait = False

        # PID gains
        # self.kp = [4, 5, 4, 4, 4, 4, 4]
        self.kp = [3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5]
        self.ki = [0.03, 0.1, 0.03, 0.03, 0.03, 0.03, 0.03]
        self.kd = [0, 0, 0, 0, 0, 0, 0]

    def constrain(input, min_val, max_val):
        if input < min_val: return min_val
        elif input > max_val: return max_val
        else: return input

    def set_target_joint_positions(self, target_joint_positions):
        assert len(target_joint_positions) == vu.N_ARM_JOINTS, \
            'Expected target joint positions to be length {}, but it was length {} instead.'.format(len(target_joint_positions), vu.N_ARM_JOINTS)
        self._target_joint_positions = target_joint_positions

    def calculate_commands_from_feedback(self, timestamp, sensed_joint_positions):
        assert self._target_joint_positions is not None, \
            'Expected target joint positions to be set, but it was not.'

        # Using the input joint feedback, and the known target joint positions,
        # calculate the joint commands necessary to drive the system towards
        # the target joint positions
        ctrl_commands = np.zeros(vu.N_ARM_JOINTS)
        
        # Input and setpoint
        current = np.asarray(sensed_joint_positions)
        target = np.asarray(self._target_joint_positions)

        # Error term
        error = target - current
        # print(error)

        # PID loop
        if self.first_loop: 
            self.first_loop = False
        else:
            dt = timestamp - self.history['timestamp'][-1]
            ctrl_commands = np.multiply(self.kp, error)
            ctrl_commands += np.multiply(self.kd, current - np.asarray(self.history['joint_feedback'][-1]) / dt)
            ctrl_commands += np.multiply(self.ki, error)

        # Append time history
        self.history['timestamp'].append(timestamp)
        self.history['joint_feedback'].append(sensed_joint_positions)
        self.history['joint_target'].append(self._target_joint_positions)
        self.history['ctrl_commands'].append(ctrl_commands)
        
        return ctrl_commands

    def has_stably_converged_to_target(self):
        # Check convergence for last 10 states
        converged = np.all(np.isclose(self.history['joint_target'][-1],
            self.history['joint_feedback'][-10:], 1e-2, 1e-2))
        
        # First converged
        if converged and not self.converged_wait:
            self.converged_wait = True
            self.converged_time = time.time()
            return False
        
        # Wait for <wait_time> seconds before next state
        if self.converged_wait and (time.time() - self.converged_time > self.wait_time):
            self.converged_wait = False
            return True
        
        return False

    def plot(self, savefile=False):
        for i in range(vu.N_ARM_JOINTS):
            fig = plt.figure()
            plt.plot(self.history['timestamp'], np.array(self.history['joint_feedback'])[:, i])
            plt.plot(self.history['timestamp'], np.array(self.history['joint_target'])[:, i])
            plt.legend(labels=['Joint Sensed', 'Joint Target'], title='Legend', loc=0, fontsize='small', fancybox=True)
            plt.title('Time Response for Joint %d' % (i + 1))
            plt.xlabel('Timestamp (s)')
            
            if i > 4: plt.ylabel('Joint Displacement (m)')
            else: plt.ylabel('Joint Angle (rad)')
            
            if savefile: fig.savefig('joints/joint-%d.jpg' % (i + 1), dpi=480, bbox_inches='tight')
            else: plt.show()

    def execute(self, joint_targets, gripper_targets):
        # Iterate through target joint positions
        for target in np.hstack((joint_targets, gripper_targets)):
            # Set new target position
            self.set_target_joint_positions(target)

            steady_state_reached = False
            while not steady_state_reached:
                timestamp = vu.get_sim_time_seconds(self.clientID)
                # print('Simulation time: {} sec'.format(timestamp))

                # Get current joint positions
                sensed_joint_positions = vu.get_arm_joint_positions(self.clientID)

                # Calculate commands
                commands = self.calculate_commands_from_feedback(timestamp, sensed_joint_positions)

                # Send commands to V-REP
                vu.set_arm_joint_target_velocities(self.clientID, commands)

                # Print current joint positions (comment out if you'd like)
                # print(sensed_joint_positions)
                vu.step_sim(self.clientID, 1)

                # Determine if we've met the condition to move on to the next point
                steady_state_reached = self.has_stably_converged_to_target()


def main(args):
    success = False
    try:
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

        # Joint targets, radians for revolute joints and meters for prismatic joints
        gripper_targets = np.asarray([[-0.03, 0.03], [-0.03, 0.03]])
        joint_targets = np.radians([[-80, 0, 0, 0, 0], [0, 60, -75, -75, 0]])

        # Verify fk
        from utils import URDFRobot, print_pose
        robot = URDFRobot('urdf/locobot_description_v3.urdf')

        from forward_kinematics import ForwardKinematicsSolver
        fksolver = ForwardKinematicsSolver(robot)
        for target in joint_targets:
            print(target)
            pose = fksolver.getWristPose(target[:5], robot, vu.ARM_JOINT_NAMES[:5])
            print_pose(pose)

        # Instantiate controller and run it
        controller = ArmController(clientID)
        controller.execute(joint_targets, gripper_targets)
        success = True

    except Exception as e:
        print(e)
        traceback.print_exc()

    finally:
        # Stop VREP simulation
        vu.stop_sim(clientID)

    # Plot time histories
    if success: controller.plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
