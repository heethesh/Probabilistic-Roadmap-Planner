# Python 2/3 compatibility
from __future__ import print_function

import os
import sys

import numpy as np
from urdf_parser_py.urdf import URDF
from tf.transformations import euler_matrix, quaternion_matrix, rotation_matrix, translation_matrix, decompose_matrix

from cuboid_collision import Cuboid
from utilities import vrep_utils as vu

# Paths
PKG_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Handle names
LINK_NAMES = ['arm_base_link_joint', 'shoulder_link', 'elbow_link', 'forearm_link', 'wrist_link', 'gripper_link', 'finger_r', 'finger_l']
JOINT_NAMES = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']
OBSTACLE_NAMES = ['cuboid_0', 'cuboid_1', 'cuboid_2', 'cuboid_3', 'cuboid_4', 'cuboid_5']


def block_printing(func):
    def func_wrapper(*args, **kwargs):
        # Block all printing to the console
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

        # Function call
        value = func(*args, **kwargs)
        
        # Enable all printing to the console
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        return value
    return func_wrapper


def print_pose(pose):
    _, _, R, T, _ = decompose_matrix(pose)
    print('R:', np.degrees(R), 'T:', T, '\n')
    return R, T


def convert_directed_to_undirected_graph(graph):
    undirected_graph = graph.copy()
    for node in graph:
        for child, weight in graph[node].items():
            if child not in graph or node not in graph[child]:
                undirected_graph.setdefault(child, {})[node] = weight
    return undirected_graph


class URDFRobot:
    def __init__(self, filename):
        self.robot = self.parse_urdf(filename)

    @block_printing
    def parse_urdf(self, filename):
        return URDF.from_xml_file(os.path.join(PKG_PATH, filename))

    def find_joint_urdf(self, name):
        for joint in self.robot.joints:
            if joint.name == name:
                return joint
        return None

    def get_joint_axis(self, name):
        joint = self.find_joint_urdf(name)
        return np.asarray(joint.axis)

    def get_joint_limits(self, name):
        joint = self.find_joint_urdf(name)
        return (joint.limit.lower, joint.limit.upper)

    def get_joint_pose(self, name):
        joint = self.find_joint_urdf(name)
        T = translation_matrix(joint.origin.xyz)
        r, p, y = joint.origin.rpy
        R = euler_matrix(r, p, y)
        axis = np.asarray(joint.axis)
        return R, T, axis


def get_joint_pose_vrep(clientID, joint_names=None):
    if not joint_names:
        joint_names = JOINT_NAMES

    poses = []
    for handle in [vu.get_handle_by_name(clientID, j) for j in joint_names]:
        # Query VREP to get the parameters
        origin = vu.get_object_position(clientID, handle)
        rotation = vu.get_object_orientation(clientID, handle)
        poses.append((origin, rotation))

    return poses


def get_cuboids(clientID, names, append=True):
    if append: handles = [(vu.get_handle_by_name(clientID, j + '_collision_cuboid'), j) for j in names]
    else: handles = [(vu.get_handle_by_name(clientID, j), j) for j in names]

    cuboids = []
    for (handle, name) in handles:
        # Query VREP to get the parameters
        min_pos, max_pos = vu.get_object_bounding_box(clientID, handle)
        origin = vu.get_object_position(clientID, handle)
        rotation = vu.get_object_orientation(clientID, handle)

        # Cuboid parameters
        min_pos = np.asarray(min_pos)
        max_pos = np.asarray(max_pos)
        dxyz = np.abs(max_pos - min_pos)

        # Cuboid objects
        cuboids.append(Cuboid(origin, rotation, dxyz, name, vrep=True))

    return cuboids

if __name__ == '__main__':
    robot = URDFRobot('urdf/locobot_description_v3.urdf')
    print(robot.get_joint_limits('joint_1'))
