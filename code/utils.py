# Python 2/3 compatibility
from __future__ import print_function

import os
import sys

import numpy as np

from utilities import vrep_utils as vu
from cuboid_collision import Cuboid, CollisionChecker


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


def get_cuboids(clientID, names, append=True):
    if append: handles = [vu.get_handle_by_name(clientID, j + '_collision_cuboid') for j in names]
    else: handles = [vu.get_handle_by_name(clientID, j) for j in names]

    cuboids = []
    for handle in handles:
        min_pos, max_pos = vu.get_object_bounding_box(clientID, handle)
        origin = vu.get_object_position(clientID, handle)
        rotation = vu.get_object_quaternion(clientID, handle)

        # Cuboid parameters
        min_pos = np.asarray(min_pos)
        max_pos = np.asarray(max_pos)
        dxyz = np.abs(max_pos - min_pos)

        # Cuboid objects
        cuboids.append(Cuboid(origin, rotation, dxyz))

    return cuboids
