import numpy as np
from math import cos, sin, pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gymnasium as gym  # original gym is no longer maintained and now called gymnasium


class RobotEnvironment(gym.Env):
    def __init__(self,  radius=0.03, height_per_turn=0.05, turns=2, resolution=0.001):
        # each joint can have one of three actions: decrease (-0.1°), keep (0.0°), increase (+0.1°)
        # represented as 0 (decrease), 1 (keep), 2 (increase) for each join
        self.action_space =  gym.spaces.MultiDiscrete([3, 3, 3, 3, 3, 3])
        
        # define the helix
        self.radius = radius  # radius in meters
        self.height_per_turn = height_per_turn  # height per turn in meters
        self.turns = turns  # number of turns

        # voxel space dimensions
        x_range = (-self.radius, self.radius)
        y_range = (-self.radius, self.radius)
        z_range = (0, self.height_per_turn*self.turns)
        self.resolution = resolution  # resolution: 1mm

        # dimensions of the voxel grid
        x_size = int((x_range[1] - x_range[0]) / self.resolution) + 1
        y_size = int((y_range[1] - y_range[0]) / self.resolution) + 1
        z_size = int((z_range[1] - z_range[0]) / self.resolution) + 1

        # observation space defined by dimensions and possible voxel values (-1 to 1)
        self.observation_space = gym.spaces.Box(low=-1, high=1, 
                                                shape=(x_size, y_size, z_size), 
                                                dtype=np.int8)

        # init voxel space 
        self.voxel_space = np.full((x_size, y_size, z_size), -1)  # initialize all voxels with -1

        # Populate the voxel space with a helix
        self.init_helix()

        # Define the initial state of the robot
        self.start_position = np.array([self.radius, 0, 0])  # Starting position of the TCP
        self.joint_angles = np.array([0, 0, 0, 0, 0, 0])  # Initial joint angles
        self.tcp_position = self.forward_kinematics(self.joint_angles)  # Initial end-effector position
        self.tcp_on_helix = self.is_on_helix(self.tcp_position, voxel_space, resolution, x_range, y_range, z_range)  # Check if the TCP is on the helix
        
        self.reward = 0 # reward points
        self.reached_target = False

    def init_helix(self):
        # initialize helix
        r = 0.03  # radius 
        h = 0.05  # height per turn 
        t = np.linspace(0, 2, 100)  # parameter t from 0 to 2 for 2 complete turns
        helix_x = r * np.cos(2 * np.pi * t)
        helix_y = r * np.sin(2 * np.pi * t)
        helix_z = h * t

        # mark the voxels on the helix path:
        for i in range(len(helix_x)):
            x_idx = int(round((helix_x[i] - self.x_range[0]) / self.resolution))
            y_idx = int(round((helix_y[i] - self.y_range[0]) / self.resolution))
            z_idx = int(round((helix_z[i] - self.z_range[0]) / self.resolution))
            if i == len(helix_x) - 1:  # last helix point
                self.voxel_space[x_idx, y_idx, z_idx] = 1
            else:
                self.voxel_space[x_idx, y_idx, z_idx] = 0  # helix path


    def step(self, action):
        # apply action, update the state of the environment, return observation, etc.
        pass

    def reset(self):
        # reset the environment to an initial state and return the initial observation
        self.voxel_space.fill(-1)  # reset the voxel space
        self.init_helix()  # reinitialize the helix
        return self.voxel_space

    def render(self, mode='human', tcp_coords=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*np.where(self.voxel_space == 1), c='r', s=50, alpha=1)
        ax.scatter(*np.where(self.voxel_space == 0), c='b', s=50, alpha=1)
        
        # Optional: Highlight a specific TCP position if provided
        if tcp_coords is not None:
            tcp_value = self.is_on_helix(tcp_coords)
            if tcp_value != -1:
                tcp_x, tcp_y, tcp_z = tcp_coords
                ax.scatter([tcp_x], [tcp_y], [tcp_z], c='g', s=100, label='TCP Position')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D-Plot of the Voxel Space')
        plt.show()


    def process_action(self, action):
        # convert action indices (0, 1, 2) to deltas (-0.1, 0.0, +0.1 degrees)
        delta_angles = [(a - 1) * 0.1 for a in action]
        return delta_angles


    def dh_transform_matrix(self,a, d, alpha, theta):
    
    ## compute the Denavit-Hartenberg transformation matrix.
    
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

        # extract position from the final transformation matrix
        position = T[:3, 3]
        return position
    
    def reward_function(self,tcp_on_helix):
        if tcp_on_helix:
            self.reward += 1
        else:
            self.reward -=1 
           
    
    def is_on_helix(self, tcp_coords):
        x_idx = int(round((tcp_coords[0] - self.x_range[0]) / self.resolution))
        y_idx = int(round((tcp_coords[1] - self.y_range[0]) / self.resolution))
        z_idx = int(round((tcp_coords[2] - self.z_range[0]) / self.resolution))

        if 0 <= x_idx < self.voxel_space.shape[0] and 0 <= y_idx < self.voxel_space.shape[1] and 0 <= z_idx < self.voxel_space.shape[2]:
            voxel_value = self.voxel_space[x_idx, y_idx, z_idx]
            if voxel_value == 1:
                self.reached_target = True
                return True
            elif voxel_value == 0:
                return True
        return False

