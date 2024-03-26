### ###############################################################################################################
### Reinforcement Learning - WS 2023/24 
### 6-joint robot arm
### Group: Dennis Huff, Philip, Ari Wahl 
### ###############################################################################################################

# https://gymnasium.farama.org/api/env/


import os

# mute the MKL warning on macOS
os.environ["MKL_DEBUG_CPU_TYPE"] = "5"

import numpy as np
from math import cos, sin, pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gymnasium as gym  # original gym is no longer maintained and now called gymnasium


class RobotEnvironment(gym.Env):
    """Custom Environment that follows gym interface"""
    """Every Gym environment must have the attributes action_space and observation_space (containing states). """

    def __init__(self,  radius=0.03, height_per_turn=0.05, turns=2, resolution=0.001):
        # each joint can have one of three actions: decrease (-0.1°), keep (0.0°), increase (+0.1°)
        # represented as 0 (decrease), 1 (keep), 2 (increase) for each join
        self.action_space =  gym.spaces.MultiDiscrete([3, 3, 3, 3, 3, 3])  # 3^6 = 729 possible actions
        
        # define the helix
        self.radius = radius  # radius in meters
        self.height_per_turn = height_per_turn  # height per turn in meters
        self.turns = turns  # number of turns

        # voxel space dimensions
        self.x_range = (-self.radius, self.radius)
        self.y_range = (-self.radius, self.radius)
        self.z_range = (0, self.height_per_turn*self.turns)
        self.resolution = resolution  # resolution: 1mm

        # dimensions of the voxel grid
        x_size = int((self.x_range[1] - self.x_range[0]) / self.resolution) + 1
        y_size = int((self.y_range[1] - self.y_range[0]) / self.resolution) + 1
        z_size = int((self.z_range[1] - self.z_range[0]) / self.resolution) + 1

        # observation space defined by dimensions and possible voxel values (-1 to 1)
        self.observation_space = gym.spaces.Box(low=-1, high=1, 
                                                shape=(x_size, y_size, z_size), 
                                                dtype=np.int8)
        print("Observation Space:", self.observation_space)
        print("  - Dimensionen:", self.observation_space.shape)
        print("  - low value:", self.observation_space.low)
        print("  - high value:", self.observation_space.high)
        print("  - datatype:", self.observation_space.dtype)

        # init voxel space 
        self.voxel_space = np.full((x_size, y_size, z_size), -1)  # initialize all voxels with -1
        print("Voxel Space Shape:", self.voxel_space.shape) # Voxel Space Shape: (61, 61, 101)
        print("Voxel Space Size flattened:", len(self.voxel_space.flatten()) )# Voxel Space Size: 375821

        # create a separate matrix to store the helix path
        self.helix_path = np.full_like(self.voxel_space, -1, dtype=np.int8)
        #print("observation Space Size flattened:", len(self.observation_space.flatten()) )
        # Populate the voxel space with a helix
        self.init_helix()

        # Define the initial state of the robot
        # set initial TCP position at the start of the helix
        self.tcp_position = np.array([self.radius, 0, 0], dtype=np.float64)  # fixed to the starting point of the helix on z=0
        self.joint_angles = np.array([90, 90, 180, 62.14, -150.67 ,0])   # see output of find_starting_joint_angles.py

        #self.tcp_position = self.forward_kinematics(self.joint_angles)  # initial end-effector position
        self.tcp_on_helix = self.is_on_helix(self.tcp_position)  # is the TCP is on the helix?
        
        self.reward = 0 # reward points
        self.terminated = False
        self.truncated = False


    def step(self, action):
        """Updates an environment with actions returning the next agent observation, 
        the reward for taking that actions,
        if the environment has terminated or truncated due to the latest action 
        and information from the environment about the step, i.e. metrics, debug info.

        Args:
            action (np.array): action provided by the agent (array of 6 integers between 0 and 2)

        Returns:
            observation (state):  An element of the environment’s observation_space as the next observation due to the agent actions
            reward (float): Amount of reward due to the agent actions
            terminated (bool): A boolean, indicating whether the episode has ended successfully
            truncated (bool): A boolean, indicating whether the episode has ended prematurely
            info (dict): A dictionary containing other diagnostic information from the environment
        """
        # convert action to delta angles and apply them
        delta_angles = self.process_action(action)

        # Convert delta_angles to numpy array
        delta_angles = np.array(delta_angles)

        # Ensure delta_angles has the same dtype as self.joint_angles
        delta_angles = delta_angles.astype(self.joint_angles.dtype)

        # Check if the angle is within the correct range of -180 to 180 degrees
        if self.joint_angles > 180:
            self.joint_angles = 180
        elif self.joint_angles < -180:
            self.joint_angles = -180
        else:
            # update joint angles
            self.joint_angles += delta_angles

        # update TCP position (based on the new joint angles)
        self.tcp_position = self.forward_kinematics(self.joint_angles)

        # is TCP on the helix?
        self.tcp_on_helix = self.is_on_helix(self.tcp_position)

        # update the reward (based on the new state)
        self.reward, self.terminated, self.truncated = self.reward_function(self.tcp_on_helix)

        # eventually also return an info dictionary (for debugging)
        info = {
            'robot_state': self.joint_angles.tolist(),
            'tcp_position': self.tcp_position.tolist()
        }

        # has to return: new observation (state), reward, terminated(bool), truncated(bool) info(dict)
        # return the new observation (state), reward, done flag
        return self.voxel_space.flatten(), self.reward, self.terminated, self.truncated, info




    def reward_function(self, tcp_on_helix):
        """Calculate the reward based on the current state of the environment."""
        # initialize reward, terminated, and truncated flags
        if tcp_on_helix:
            self.reward += 1
            self.truncated = False

        if self.reached_target:
            self.reward += 100 # extra reward for reaching the target
            self.terminated = True

        else:
            truncated = True # terminate the episode if the tcp is not on the helix any more
            self.reward -=10
           
        return self.reward, self.terminated, self.truncated
    
    #def init_helix(self):
    #    # initialize helix
    #    r = 0.03  # radius 
    #    h = 0.05  # height per turn 
    #    t = np.linspace(0, 2, 100)  # parameter t from 0 to 2 for 2 complete turns
    #    helix_x = r * np.cos(2 * np.pi * t)
    #    helix_y = r * np.sin(2 * np.pi * t)
    #    helix_z = h * t
#
    #    # mark the voxels on the helix path:
    #    for i in range(len(helix_x)):
    #        x_idx = int(round((helix_x[i] - self.x_range[0]) / self.resolution))
    #        y_idx = int(round((helix_y[i] - self.y_range[0]) / self.resolution))
    #        z_idx = int(round((helix_z[i] - self.z_range[0]) / self.resolution))
    #        if i == len(helix_x) - 1:  # last helix point
    #            self.observation_space[x_idx, y_idx, z_idx] = 1
    #            #self.voxel_space[x_idx, y_idx, z_idx] = 1
    #        else:
    #            self.observation_space[x_idx, y_idx, z_idx] = 0
    #            #self.voxel_space[x_idx, y_idx, z_idx] = 0  # helix path

    def init_helix(self):
        # # create a separate matrix to store the helix path
        #helix_path = np.full_like(self.voxel_space, -1, dtype=np.int8)

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
                self.helix_path[x_idx, y_idx, z_idx] = 1
            else:
                self.helix_path[x_idx, y_idx, z_idx] = 0  # helix path

        # assign the helix path matrix to the observation space
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=self.helix_path.shape, dtype=np.int8)
        self.observation_space = self.helix_path

    
    def is_on_helix(self, tcp_coords):
        # convert TCP coordinates to voxel indices. Therefore find the relative position of the TCP
        # within the bounds  `x_range`, `y_range`, and `z_range` 
        # scale this position to the resolution of the voxel grid:
        x_idx = int(round((tcp_coords[0] - self.x_range[0]) / self.resolution))
        y_idx = int(round((tcp_coords[1] - self.y_range[0]) / self.resolution))
        z_idx = int(round((tcp_coords[2] - self.z_range[0]) / self.resolution))

        # check if these indices are in the voxel space. If not, the TCP is outside the voxel space.
        if 0 <= x_idx < self.voxel_space.shape[0] and 0 <= y_idx < self.voxel_space.shape[1] and 0 <= z_idx < self.voxel_space.shape[2]:
            # get value of the voxel at the calculated indices. check if voxel
            # is on the helix path (0), the target/end of the helix (1), or outside the helix voxels (-1).
            voxel_value = self.voxel_space[x_idx, y_idx, z_idx]
            
            # if the TCP has reached the target (voxel-value = 1):
            if voxel_value == 1:
                self.terminateed = True
                return True  # TCP is on the helix 
            
            # TCP is on a voxel of helix path but has not yet reached the end yet (voxel-value = 0):
            elif voxel_value == 0:
                return True # TCP is on the helix path

        # otherwise the TCP is not on the helix path any more
        return False



    def reset(self):
        """Resets the environment to an initial internal state, returning an initial observation and info.

        Returns:
            observsation (state): Observation of the initial state. 
                                            This will be an element of observation_space (typically a numpy array) 
                                            and is analogous to the observation returned by step()
            info(dict): A dictionary containing additional information about the environment
                            analogous to the info returned by step()
        """
        # reset the environment 
        self.voxel_space.fill(-1)
        self.init_helix()

        # reset the joint angles and TCP position to the start of the helix
        self.tcp_position = np.array([self.radius, 0, 0], dtype=np.float64)  # fixed to the starting point of the helix on z=0
        self.joint_angles = np.array([90, 90, 180, 62.14, -150.67 ,0])   # see output of find_starting_joint_angles.py

        # reset the reward and Flags
        self.tcp_on_helix = self.is_on_helix(self.tcp_position)
        self.reward = 0
        self.terminated= False
        self.truncated = False

        state = self.voxel_space.flatten()   # flatten besser hier oder im netzwerk?
        # todo: return first observation from observation_space

        # Check the size of the state vector
        expected_size = 1 * 61  * 61 * 101
        actual_size = state.size
        print("Actual size of state vector:", actual_size)
        print("Expected size of state vector:", expected_size)

        # If the sizes don't match, print an error message and return None
        if actual_size != expected_size:
            print("Error: Size of state vector doesn't match the expected size.")
            return None

        # Reshape the state vector
        state = np.reshape(state, (1, expected_size)) # davor 61
        print("State: ", state)

        # eventually also return an info dictionary (for debugging)
        info = {
            'robot_state': self.joint_angles.tolist(),
            'tcp_position': self.tcp_position.tolist()
        }

        return state, info
    
    # def reset(self):
    #     # reset the environment 
    #     self.voxel_space.fill(-1)
    #     self.init_helix()

    #     # reset the joint angles and TCP position
    #     self.joint_angles = np.array([0, 0, 0, 0, 0, 0])
    #     self.tcp_position = self.forward_kinematics(self.joint_angles)

    #     # reset the reward and Flags
    #     self.tcp_on_helix = True
    #     self.reward = 0
    #     self.reached_target = False

    #     return self.voxel_space
    
    # def calculate_initial_joint_angles(self):
    #     # we don't need this if the initial joint angles are np.array([0, 0, 0, 0, 0, 0]) ---> but is this a valid option?
    #     initial_position = self.helix_points[0]  # Assuming helix_points[0] is the starting point on the helix
    #     initial_angles = solve_inverse_kinematics(initial_position)  # You need to implement this
    #     return initial_angles

    def render(self, mode='human', tcp_coords=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*np.where(self.voxel_space == 1), c='r', s=40, alpha=1)  # helix end points
        ax.scatter(*np.where(self.voxel_space == 0), c='b', s=40, alpha=1)  # helix path points
        
        ax.scatter(*np.where(self.observation_space == 1), c='r', s=50, alpha=1)  # helix end points
        ax.scatter(*np.where(self.observation_space == 0), c='b', s=50, alpha=1)
        # if TCP coordinates are provided and valid, then visualize TCP position
        if tcp_coords is not None:
            print(f"TCP Coordinates: {tcp_coords}")
            is_on_path = self.is_on_helix(tcp_coords)
            print(f"Is TCP on Helix Path: {is_on_path}")
            if is_on_path:
                # convert real-world coordinates to indices for visualization
                x_idx = (tcp_coords[0] - self.x_range[0]) / self.resolution
                y_idx = (tcp_coords[1] - self.y_range[0]) / self.resolution
                z_idx = (tcp_coords[2] - self.z_range[0]) / self.resolution
                
                # highlight TCP position
                ax.scatter([x_idx], [y_idx], [z_idx], c='lightgreen', s=100, alpha= 1, label='TCP Position')

        ax.set_xlabel('X Index')
        ax.set_ylabel('Y Index')
        ax.set_zlabel('Z Index')
        ax.set_title('3D Plot of the Voxel Space')
        plt.legend()
        plt.show()

    def process_action(self, action):
  
          # Check if action is iterable
          if isinstance(action, (list, tuple)):
          # If yes, calculate the delta angles for each action
              delta_angles = [(a - 1) * 0.1 for a in action]
          else:
          # Otherwise, there is only one action, so calculate the delta angle directly
              delta_angles = [(action - 1) * 0.1]

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
        for i, params in enumerate(dh_params):
            a, d, alpha, theta_val = params
            print(f"DH Parameters for joint {i}: a={a}, d={d}, alpha={alpha}, theta={theta_val}")

            T_i = self.dh_transform_matrix(*params)
            print(f"Transformation Matrix T{i}:\n{T_i}\n")

            # Überprüfe die Form der Transformationsmatrix
            if T_i.shape != (4, 4):
                raise ValueError(f"Unexpected shape of transformation matrix T{i}: {T_i.shape}. Expected (4, 4)")

            T = np.dot(T, T_i)
        print(f"Final Transformation Matrix T:\n{T}\n")  # Neu hinzugefügter Code

        # for params in dh_params:
        #     T = np.dot(T, self.dh_transform_matrix(*params))

        # extract position from the final transformation matrix
        position = T[:3, 3]
        return position
    
if __name__ == "__main__":
    env = RobotEnvironment()
    env.render(tcp_coords=env.tcp_position)
