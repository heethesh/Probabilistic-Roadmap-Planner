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
import sys
import time
import pickle
import pprint
import traceback

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

import utils
import locobot_joint_ctrl
from controller import ArmController
from utilities import vrep_utils as vu
from cuboid_collision import CollisionChecker
from forward_kinematics import ForwardKinematicsSolver

# Pretty printer
pp = pprint.PrettyPrinter(indent=4)


class ProbabilisticRoadMap:
    def __init__(self, clientID, joint_targets, gripper_targets, urdf,
        save_graph=True, load_graph=False):
        
        # VREP client ID
        self.clientID = clientID

        # Targets
        self.joint_targets = { 'start': joint_targets[0], 'end': joint_targets[1] }
        self.gripper_targets = { 'start': gripper_targets[0], 'end': gripper_targets[1] }

        # PRM variables
        self.roadmap = {}
        self.kdtree = None
        self.save_graph = save_graph
        self.load_graph = load_graph

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

    def plan(self, N=10, K=3, verbose=False):
        # Create the roadmap graph
        if self.load_graph:
            print('\nLoading roadmap graph and KD tree...')
            with open(r'graph.pickle', 'rb') as file:
                self.kdtree, self.roadmap = pickle.load(file)
            print('Done')
        else: self.construct_roadmap(N=N, K=K, verbose=verbose)

        # Query the roadmap and plan a path
        if self.query_roadmap(K=K, verbose=verbose):
            joint_plan = [ self.kdtree.data[waypoint] for waypoint in self.path ]
            joint_plan = np.asarray([ self.joint_targets['start'] ] + joint_plan + [ self.joint_targets['end'] ])
            gripper_plan = np.linspace(self.gripper_targets['start'], self.gripper_targets['end'], num=len(self.path) + 2)

            # Return the trajectory
            print('\nJoint plan:')
            print(joint_plan)
            return True, joint_plan, gripper_plan

        # Failed to plan a trajectory
        else:
            return False, None, None

    ############### Roadmap construction ###############

    def construct_roadmap(self, N, K, verbose=False):
        # Random sampling
        print('\nRandom sampling...')
        nodes = []
        while len(nodes) < N:
            sample = self.get_random_sample()

            # Check is sampled point is in collision and above ground
            if self.collision_free(sample, verbose=verbose):
                nodes.append(sample)

        # Create a KDTree
        nodes = np.asarray(nodes)
        self.kdtree = KDTree(nodes)

        # Construct roadmap graph
        print('\nConstructing roadmap graph...')
        for i, node in enumerate(nodes):
            # Find k-nearest neighbors
            [distances], [args] = self.kdtree.query(node.reshape(1, -1), k=K + 1)
            if verbose: print('%d NN:' % K, i, args[1:])

            # Run local planner between current node and all nearest nodes
            for arg, dist in zip(args[1:], distances[1:]):
                if self.local_planner(node, nodes[arg]):
                    self.roadmap.setdefault(i, {})[arg] = dist
        
        # Save KDTree and roadmap
        if self.save_graph:
            print('\nSaving roadmap graph and KD tree...')
            with open(r'graph.pickle', 'wb') as file:
                pickle.dump([self.kdtree, self.roadmap], file)

        print('Done')

    def get_random_sample(self):
        limits = [ self.robot.get_joint_limits(j) for j in utils.JOINT_NAMES[:5] ]
        joints = [ np.random.uniform(i, j) for (i, j) in limits ]
        return np.asarray(joints)
        
    def collision_free(self, joint_angles=None, verbose=False):
        # Compute link poses using forward kinematics
        if joint_angles is not None:
            R, T, self.link_cuboids = self.fksolver.compute(self.link_cuboids, joint_angles)
        
        # Collision check between obstacles and arm
        for link in self.link_cuboids[1:]:
            for obstacle in self.obstacle_cuboids:
                if self.collision_checker.detect_collision_optimized(link, obstacle):
                    if verbose: print('Collision between "%s" and "%s"' % (link.name, obstacle.name))
                    return False

        # Self-collision check
        for i, link_a in enumerate(self.link_cuboids[1:]):
            for j, link_b in enumerate(self.link_cuboids[1:]):
                # Skip self-link, adjacent links check, and gripper-fingers check
                if (i == j) or (i == j + 1) or (i == j - 1) \
                    or (i == 4 and (j in [5, 6])) or (j == 4 and (i in [5, 6])): continue
                elif self.collision_checker.detect_collision_optimized(link_a, link_b):
                    if verbose: print('Self-collision between "%s" and "%s"' % (link_a.name, link_b.name))
                    return False

        # Check if all link vertices are above ground plane
        for link in self.link_cuboids[1:]:
            if np.sum(link.vertices[:, -1].flatten() < 0):
                if verbose: print('Link "%s" is below ground plane' % link.name)
                return False

        # All checks passed
        return True

    def local_planner(self, start, end, n_points=5, verbose=False):
        # Check dimensions
        assert len(start) == len(end)

        # Interpolate to form a local path
        local_path = np.linspace(start, end, num=n_points)
        
        # Check for collisions along the path
        for point in local_path:
            if not self.collision_free(point, verbose=verbose):
                return False

        # Path planning successful
        return True

    ############### Roadmap query ###############

    def query_roadmap(self, K=3, verbose=False):
        # Add start and end targets to the roadmap
        start = self.joint_targets['start']
        end = self.joint_targets['end']
        self.add_targets_to_roadmap(start, end, K=K, verbose=verbose)
        self.roadmap = utils.convert_directed_to_undirected_graph(self.roadmap)
        
        # Print roadmap
        if verbose: 
            print('\nRoadmap:')
            pp.pprint(self.roadmap)
        self.display_roadmap()

        # Run the path planner
        return self.path_planner(verbose=verbose)

    def add_targets_to_roadmap(self, start, end, K, verbose=False):
        print('\nAdding targets to roadmap...')

        # Pick k-nearest neighbors for start and end nodes
        [start_dists], [start_args] = self.kdtree.query(start.reshape(1, -1), k=K)
        [end_dists], [end_args] = self.kdtree.query(end.reshape(1, -1), k=K)

        # Display target KNNs
        if verbose:
            print('%d NN: START' % K, start_args[1:])
            print('%d NN: END' % K, end_args[1:])

        # Connect start node to knn nodes in the roadmap
        for dist, arg in zip(start_dists, start_args):
            if self.local_planner(start, self.kdtree.data[arg]):
                self.roadmap.setdefault('start', {})[arg] = dist

        # Connect end node to knn nodes in the roadmap
        for dist, arg in zip(end_dists, end_args):
            if self.local_planner(end, self.kdtree.data[arg]):
                self.roadmap.setdefault('end', {})[arg] = dist

        print('Done')

    def path_planner(self, verbose=False):
        # Initialize planner
        path = []
        parents = {}
        unvisited = self.roadmap

        # Initialize weights
        weights = { node: np.Inf for node in unvisited }
        weights['start']  = 0
        weights['end'] = np.Inf

        # Dijkstra's path planner
        print('\nRunning global planner...')
        while unvisited:
            # Find the nearest node (based on norm from KDTree)
            nearest = None
            for node in unvisited:
                if nearest is None or (nearest in self.roadmap and weights[node] < weights[nearest]):
                    nearest = node

            # Explore children nodes
            exists = nearest in self.roadmap
            for child, weight in self.roadmap[nearest].items():
                if exists and weight + weights[nearest] < weights[child]:
                    # Update weights
                    weights[child] = weight + weights[nearest]
                    # Keep track of parent node
                    parents[child] = nearest

            # Mark the node visited
            unvisited.pop(nearest)

        if verbose:
            print('\nWeights:')
            pp.pprint(weights)
            print('\nParents:')
            pp.pprint(parents)

        # Backtrack to start node
        cur_node = 'end'
        while True:
            # Reached start node
            if cur_node == 'start':
                path.reverse()
                self.path = path[:-1]
                break

            # Backtrack to parent node
            path.append(cur_node)
            try: cur_node = parents[cur_node]
            except KeyError:
                print('Failed to find a path! Try making the graph more dense.')
                return False

        print('Done')
        return True

    ############### Visualizations ###############

    def display_cuboids(self, cuboids=None, obstacles=False, return_ax=False):
        # Display collision cuboids
        if cuboids:
            return self.collision_checker.display_cuboids(cuboids, return_ax=return_ax)
        elif obstacles:
            return self.collision_checker.display_cuboids(self.link_cuboids + self.obstacle_cuboids, return_ax=return_ax)
        else:
            return self.collision_checker.display_cuboids(self.link_cuboids, return_ax=return_ax)

    def display_roadmap(self, title=None, savefile=None):
        # Get cuboid display figure object
        ax = self.display_cuboids(self.obstacle_cuboids, return_ax=True)

        # Visualize the nodes in the roadmap
        for key in self.roadmap:
            # Get Cartesian space coordinates of root node
            if type(key) is int:
                try: key_pos = self.fksolver.compute(self.link_cuboids, self.kdtree.data[key])[1]
                except TypeError:
                    print('Skipping display of %s children' % key)
                    continue
            else: key_pos = self.fksolver.compute(self.link_cuboids, self.joint_targets[key][:5])[1]

            # Get Cartesian space coordinates of child nodes
            values_pos = []
            for val in self.roadmap[key]:
                try: values_pos.append(self.fksolver.compute(self.link_cuboids, self.kdtree.data[val])[1])
                except TypeError:
                    print('Skipping display of node %s\'s children' % key)
                    continue

            # Label 3D coordinates
            ax.text(key_pos[0], key_pos[1], key_pos[2], key if type(key) is str else str(key))

            # Display edges
            colors = plt.get_cmap('tab10')
            for val_pos in values_pos:
                color = colors(key % 10) if type(key) is int else 'r'
                marker = 'o' if type(key) is int else 'x'
                ax.plot([key_pos[0], val_pos[0]], [key_pos[1], val_pos[1]], [key_pos[2], val_pos[2]],
                    marker=marker, color=color, linestyle='-')

        if title: plt.title(title)
        if savefile: fig.savefig('%s.jpg' % savefile, dpi=480, bbox_inches='tight')
        else: plt.show()


def setup_simulator():
    # Connect to V-REP
    print ('\nConnecting to V-REP...')
    clientID = vu.connect_to_vrep()
    print ('Connected.')

    # Reset simulation in case something was running
    vu.reset_sim(clientID)

    # Initialize control inputs
    vu.set_arm_joint_target_velocities(clientID, np.zeros(vu.N_ARM_JOINTS))
    vu.set_arm_joint_forces(clientID, 50.*np.ones(vu.N_ARM_JOINTS))
    vu.step_sim(clientID, 1)

    return clientID


def main():
    success = False
    try:
        # Setup and reset simulator
        clientID = setup_simulator()

        # Joint targets, radians for revolute joints and meters for prismatic joints
        joint_targets = np.radians([[-80, 0, 0, 0, 0], [0, 60, -75, -75, 0]])
        gripper_targets = np.asarray([[-0.03, 0.03], [-0.03, 0.03]])
        
        # Initiate PRM planner
        prm_planner = ProbabilisticRoadMap(clientID, joint_targets, gripper_targets,
            urdf='urdf/locobot_description_v3.urdf', load_graph=True)

        # Run planner
        ret, joint_plan, gripper_plan = prm_planner.plan(N=100, K=10, verbose=False)

        # Planning successful
        if ret:
            # Setup and reset simulator
            clientID = setup_simulator()

            # Instantiate controller and execute the planned trajectory
            controller = ArmController(clientID)
            controller.execute(joint_plan, gripper_plan)
            success = True

    except Exception as e:
        print(e)
        traceback.print_exc()

    finally:
        # Stop VREP simulation
        vu.stop_sim(clientID)

    # Plot time histories
    if success: controller.plot(savefile=True)


if __name__ == '__main__':
    main()
