# Python 2/3 compatibility
from __future__ import print_function

import operator
import itertools

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.core.umath_tests import inner1d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from tf.transformations import euler_matrix, rotation_matrix, translation_matrix


class Cuboid:
    def __init__(self, origin, rpy, dxyz):
        # Check dimensions
        assert len(origin) == 3
        assert len(rpy) == 3
        assert len(dxyz) == 3

        # Origin
        self.origin = np.asarray(origin)
        self.origin_matrix = translation_matrix(origin)

        # Orientation
        self.rpy = np.asarray(rpy)
        self.rotation_matrix = euler_matrix(rpy[0], rpy[1], rpy[2])

        # Transformation
        self.transform_matrix = np.matmul(self.origin_matrix, self.rotation_matrix)

        # Dimensions
        self.dxyz = np.asarray(dxyz)
        self.xdim = dxyz[0]
        self.ydim = dxyz[1]
        self.zdim = dxyz[2]

        # Save vertices
        self.vertices = self.get_vertices()

    def get_vertices(self):
        ops = list(itertools.product([operator.add, operator.sub], repeat=3))
        vertices = [(op[0](0, self.xdim / 2.0), 
                     op[1](0, self.ydim / 2.0),
                     op[2](0, self.zdim / 2.0), 1) for op in ops ]
        vertices = np.matmul(self.transform_matrix, np.asarray(vertices).T).T[:, :3]
        return vertices


def display_cuboids(cuboids, title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')

    # Equalize display aspect ratio for all axes
    points = np.asarray([0, 0, 0])
    for cuboid in cuboids: points = np.vstack((points, cuboid.vertices))
    max_range = (np.array([points[:, 0].max() - points[:, 0].min(), 
                           points[:, 1].max() - points[:, 1].min(),
                           points[:, 2].max() - points[:, 2].min()]).max() / 2.0)
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    edges = lambda x: [[x[0], x[1], x[3], x[2]], [x[4], x[5], x[7], x[6]],
                       [x[0], x[1], x[5], x[4]], [x[2], x[3], x[7], x[6]],
                       [x[0], x[2], x[6], x[4]], [x[5], x[7], x[3], x[1]]]
    
    colors = plt.get_cmap('tab10')
    for i, cuboid in enumerate(cuboids):
        ax.add_collection3d(Poly3DCollection(edges(cuboid.vertices), linewidths=2, edgecolors='k', alpha=0.5, facecolor=colors(i % 10)))
        ax.scatter(cuboid.vertices[:, 0], cuboid.vertices[:, 1], cuboid.vertices[:, 2], c='r')

    ax.set_xlabel('X-Axis')
    ax.set_ylabel('Y-Axis')
    ax.set_zlabel('Z-Axis')
    plt.title(title)
    fig.savefig('%s.jpg' % title, dpi=480, bbox_inches='tight')
    # plt.show()


def get_normals(cub1, cub2):
    normals = []
    for i in range(3):
        # Cube normals (x6)
        normals.append(cub1.rotation_matrix[:3, i])
        normals.append(cub2.rotation_matrix[:3, i])
        
        # Normals of normals (x9)
        normals += np.array_split((np.cross(cub1.rotation_matrix[:3, i], 
            cub2.rotation_matrix[:3, :3]).flatten()), 3)
        
    return np.asarray(normals)


def detect_collision(cub1, cub2):
    # Caluclate all 15 normals
    normals = get_normals(cub1, cub2)
    for normal in normals:
        # Calculate projections
        projects1 = inner1d(normal, cub1.vertices)
        projects2 = inner1d(normal, cub2.vertices)
        
        # Gap detected
        if np.max(projects1) < np.min(projects2) or \
           np.max(projects2) < np.min(projects1): return False

    return True


def run_test_cases():
    # Reference cuboid
    cuboid_ref = Cuboid([0, 0, 0], [0, 0, 0], [3, 1, 2])

    # Test cuboids
    test_cuboids = [Cuboid([0, 1, 0], [0, 0, 0], [0.8, 0.8, 0.8]),
                    Cuboid([1.5, -1.5, 0], [1, 0, 1.5], [1, 3, 3]),
                    Cuboid([0, 0, -1], [0, 0, 0], [2, 3, 1]),
                    Cuboid([3, 0, 0], [0, 0, 0], [3, 1, 1]),
                    Cuboid([-1, 0, -2], [.5, 0, 0.4], [2, 0.7, 2]),
                    Cuboid([1.8, 0.5, 1.5], [-0.2, 0.5, 0], [1, 3, 1]),
                    Cuboid([0, -1.2, 0.4], [0, 0.785, 0.785], [1, 1, 1]),
                    Cuboid([-0.8, 0, -0.5], [0, 0, 0.2], [1, 0.5, 0.5])]

    # Check collisions
    for i, test_cuboid in enumerate(test_cuboids):
        ret = detect_collision(cuboid_ref, test_cuboid)
        print('Cuboid %d Collision:' % (i + 1), ret)
        display_cuboids([cuboid_ref, test_cuboid], title='Cuboid %d Collision: %s\n' % (i + 1, ret))

if __name__ == '__main__':
    run_test_cases()
