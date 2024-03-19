import numpy as np
from math import cos, sin, pi

class RobotEnvironment:
    def __init__(self):
        self.radius = 0.03  # Radius of the helix in meters
        self.height_per_turn = 0.05  # Height per turn in meters
        self.turns = 2  # Number of turns
        self.start_position = np.array([self.radius, 0, 0])  # Starting position of the TCP

    def dh_transform_matrix(self,a, d, alpha, theta):
    
    ## Compute the Denavit-Hartenberg transformation matrix.
    
        return np.array([
            [np.cos(theta), -np.sin(theta), 0, a],
            [np.sin(theta) * np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha), -d * np.sin(alpha)],
            [np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha), np.cos(alpha), d * np.cos(alpha)],
            [0, 0, 0, 1]
        ])

    def forward_kinematics(self,theta):
        """
        Calculate the end-effector position using the provided joint angles (theta).
        """
        # DH parameters for each joint: (a, d, alpha, theta)
        dh_params = [
            (0, 0.15185, np.pi/2, theta[0]),
            (-0.24355, 0, 0, theta[1]),
            (-0.2132, 0, 0, theta[2]),
            (0, 0.13105, np.pi/2, theta[3]),
            (0, 0.08535, -np.pi/2, theta[4]),
            (0, 0.0921, 0, theta[5])
        ]

        T = np.eye(4)
        for params in dh_params:
            T = np.dot(T, self.dh_transform_matrix(*params))

        # Extract position from the final transformation matrix
        position = T[:3, 3]
        return position
    
    def reward_function(self, tcp_coordinates):
        # Implementiere die Belohnungsfunktion hier
        pass

    def is_within_trajectory(self, tcp_coordinates):
        # check if tcp is in trajectory
        x, y, z = tcp_coordinates
        t = z / (self.height_per_turn * self.turns) # aktuelle Höhe TCP --> z-Ebene / durch gesamt Höhe teilen
        x_expected = self.radius * cos(2 * pi * t)
        y_expected = self.radius * sin(2 * pi * t)
        distance = np.sqrt((x - x_expected)**2 + (y - y_expected)**2)

        # tolerance?
        return distance   
    
    # hier nochmal besprechen wi edas mit voxel ist??
    def is_within_voxel(tcp_coordinates, x_range, y_range, z_range):
    
        # Check if the TCP coordinates are within the voxel space.

        x, y, z = tcp_coordinates

        if x_range[0] <= x <= x_range[1] and \
           y_range[0] <= y <= y_range[1] and \
           z_range[0] <= z <= z_range[1]:
            return True
        else:
            return False

        