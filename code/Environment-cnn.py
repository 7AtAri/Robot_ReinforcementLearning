### ###############################################################################################################
### Reinforcement Learning - WS 2023/24 
### 6-joint robot arm
### Group: Dennis Huff, Philipp Bodemann, Ari Wahl 
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
        self.y_range =  (-self.radius, self.radius)
        self.z_range = (0, self.height_per_turn*self.turns)
        self.resolution = resolution  # resolution: 1mm = 0.001m
       
        # dimensions of the voxel grid
        self.x_size = int((self.x_range[1]*2- self.x_range[0]*2) / self.resolution) + 1
        self.y_size = int((self.y_range[1] - self.y_range[0]) / self.resolution) + 1
        self.z_size = int((self.z_range[1] - self.z_range[0]) / self.resolution) + 1

        # Adjusted setup for a two-channel input
        num_channels = 2  # For voxel_space and TCP position
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(num_channels, self.x_size, self.y_size, self.z_size), dtype=np.float64)
        # observation space defined by dimensions and possible voxel values (-1 to 1)
      
        # init voxel space 
        self.voxel_space = np.full((self.x_size, self.y_size, self.z_size), -1)  # initialize all voxels with -1
        
        self.initial_joint_angles = np.array([0,0,0,0,0,0.0])  # initial joint angles
        self.initial_tcp_position = self.forward_kinematics(self.initial_joint_angles)  # initial end-effector position
        
        self.joint_angles = self.initial_joint_angles  # set joint angles to initial joint angles
        
        # voxel space origin set to the initial TCP position:
        print("Initial TCP Position:", self.initial_tcp_position)
        self.init_translation_matrix()
        # Populate the voxel space with a helix
        self.init_helix()

        self.tcp_position = self.translate_robot_to_voxel_space(self.initial_tcp_position)
        # # set the TCP position in the voxel space channel 2
        self.tcp_observation = self.embed_tcp_position(self.tcp_position)
        # stack to create a two-channel observation
        self.observation = np.stack([self.voxel_space, self.tcp_observation], axis=0)

        #self.tcp_position = self.forward_kinematics(self.joint_angles)  # initial end-effector position
        self.tcp_on_helix = self.is_on_helix(self.tcp_position)  # is the TCP is on the helix?
        print("TCP on Helix:", self.tcp_on_helix)
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
            observation (state):  stacked voxel_space and 
                                            current position of the TCP (end-effector) in the 3D space
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

        #print("Joint Angles in step:", self.joint_angles)
        # update TCP position (based on the new joint angles - not on the delta angles) 
        new_tcp_position_in_robot_space = self.forward_kinematics(self.joint_angles)  # self.joint_angles are updated in process_action
        self.tcp_position = self.translate_robot_to_voxel_space(new_tcp_position_in_robot_space)
        print( "New TCP Position:", self.tcp_position)
        # set the TCP position in the voxel space channel 2
        self.tcp_observation = self.embed_tcp_position(self.tcp_position)
        # stack to create a two-channel observation
        self.observation = np.stack([self.voxel_space, self.tcp_observation], axis=0)

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
        return self.observation, self.reward, self.terminated, self.truncated, info

    def init_translation_matrix(self):
        # self.initial_tcp_position is the initial TCP position
        # the translation vector is the negative of this position
        translation_vector = -self.initial_tcp_position
        # translation matrix:
        self.translation_matrix = np.array([
            [1, 0, 0, translation_vector[0]],
            [0, 1, 0, translation_vector[1]],
            [0, 0, 1, translation_vector[2]],
            [0, 0, 0, 1]
        ])

    def translate_robot_to_voxel_space(self, point):
        # convert the point to homogeneous coordinates for matrix multiplication
        homogeneous_point = np.append(point, 1)
        # apply the translation matrix
        translated_point_homogeneous = np.dot(self.translation_matrix, homogeneous_point)
        # convert back to cartesian coordinates
        translated_point = translated_point_homogeneous[:3]
        return translated_point
    
    def update_tcp_position_in_voxel_space(self, new_tcp_position_robot_space):
        # translate new TCP position to voxel space
        translated_position = self.translate_robot_to_voxel_space(new_tcp_position_robot_space)
        # convert translated position to voxel indices
        x_idx, y_idx, z_idx = self.position_to_voxel_indices(translated_position)
        # Update voxel grid as needed


    def reward_function(self, tcp_on_helix):
        """Calculate the reward based on the current state of the environment."""
        # initialize reward, terminated, and truncated flags
        if tcp_on_helix:
            self.reward += 10
            self.truncated = False

        if self.terminated:
            self.reward += 1000 # extra reward for reaching the target
            self.terminated = True

        else:
            self.truncated = True # terminate the episode if the tcp is not on the helix any more
            self.reward -=1
           
        return self.reward, self.terminated, self.truncated
    

    def init_helix(self):
        # # create a separate matrix to store the helix path
        #helix_path = np.full_like(self.voxel_space, -1, dtype=np.int8)

        # initialize helix
        r = self.radius  # radius 
        h = self.height_per_turn # height per turn 
        t = np.linspace(0, self.turns, num=int(self.turns*100) ) # parameter t from 0 to 2 for 2 complete turns

        offset = self.radius
        helix_x = r * np.cos(2 * np.pi * t + np.pi)  + offset# 
        helix_y = r * np.sin(2 * np.pi * t + np.pi)  
        helix_z = h * t  

        # mark the voxels on the helix path:
        for i in range(len(helix_x)):
            x_idx = int(round((helix_x[i] - self.x_range[0]) / self.resolution))
            y_idx = int(round((helix_y[i] - self.y_range[0]) / self.resolution))
            z_idx = int(round((helix_z[i] - self.z_range[0]) / self.resolution))
      
            if 0 <= x_idx < self.x_size and 0 <= y_idx < self.y_size and 0 <= z_idx < self.z_size:
                if i == len(helix_x) - 1:  # last helix point
                    self.voxel_space[x_idx, y_idx, z_idx] = 1
                else:
                    self.voxel_space[x_idx, y_idx, z_idx] = 0  # helix path
            else:
                print(f"Helix point out of bounds: {x_idx}, {y_idx}, {z_idx}")

    
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
                self.terminated = True
                return True  # TCP is on the helix 
            
            # TCP is on a voxel of helix path but has not yet reached the end yet (voxel-value = 0):
            elif voxel_value == 0:
                return True # TCP is on the helix path
        else:
            self.truncated = True
            # otherwise the TCP is not on the helix path any more
            return False



    def reset(self, seed=None, options=None):
        """Resets the environment to an initial internal state, returning an initial observation and info.

        Returns:
            observsation (state): Observation of the initial state. 
                                            This will be an element of observation_space (typically a numpy array) 
                                            and is analogous to the observation returned by step()
            info(dict): A dictionary containing additional information about the environment
                            analogous to the info returned by step()
        """
        # reset the environment 
        print("Resetting the environment...")
        _ = seed  # acknowledging the seed parameter without using it to fit the gymnasium requirements
        self.voxel_space.fill(-1)

        self.initial_joint_angles = np.array([0,0,0,0,0,0.0])  # initial joint angles
        # self.initial_tcp_position = self.forward_kinematics(self.initial_joint_angles)  # initial end-effector position
        
        # voxel space origin set to the initial TCP position:
        print("Initial TCP Position:", self.initial_tcp_position)
        self.init_translation_matrix()
        # Populate the voxel space with a helix
        self.init_helix()
        self.tcp_position = self.translate_robot_to_voxel_space(self.initial_tcp_position)
        
        # reset the joint angles and TCP position to the start of the helix
    
        self.tcp_observation = self.embed_tcp_position(self.tcp_position) # initial end-effector position

        # # set observation space to the initial state
    
        # Stack to create a two-channel observation
        self.observation = np.stack([self.voxel_space, self.tcp_observation], axis=0)

        # reset the reward and Flags
        self.tcp_on_helix = self.is_on_helix(self.tcp_position)
        self.reward = 0
        self.terminated= False
        self.truncated = False

        # eventually also return an info dictionary (for debugging)
        info = {
            'robot_state': self.initial_joint_angles.tolist(),
            'tcp_position': self.tcp_position.tolist()
        }

        return self.observation, info #  self.joint_angles  # also return the joint angles?


    def render(self, tcp_coords=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*np.where(self.voxel_space == 1), c='r', s=40, alpha=1)  # helix end points
        ax.scatter(*np.where(self.voxel_space == 0), c='b', s=40, alpha=1)  # helix path points
        
        # if TCP coordinates are provided and valid, then visualize TCP position
        if tcp_coords is not None:
            #print(f"TCP Coordinates: {tcp_coords}")
            #is_on_path = self.is_on_helix(tcp_coords)
            #print(f"Is TCP on Helix Path: {is_on_path}")

            # convert real-world coordinates to indices for visualization
            x_idx = (tcp_coords[0] - self.x_range[0]) / self.resolution
            y_idx = (tcp_coords[1] - self.y_range[0]) / self.resolution
            z_idx = (tcp_coords[2] - self.z_range[0]) / self.resolution
            
            # highlight TCP position
            ax.scatter([x_idx], [y_idx], [z_idx], c='lightgreen', s=100, alpha= 1, label='TCP Position')

        # Set axis limits to start from 0
        #ax.set_xlim(0, self.x_size)
        #ax.set_ylim(0, self.y_size)
        #ax.set_zlim(0, self.z_size)
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
            delta_angles = np.array([(a - 1) * 0.1 for a in action])
            print("Delta Angles:", delta_angles)
        else:
        # Otherwise, there is only one action, so calculate the delta angle directly
            delta_angles = np.array([(action - 1) * 0.1])
            
        print("joint_angles:", self.joint_angles)
        new_angles = self.joint_angles + delta_angles

        # Limit the new joint angles within the range of -180 to 180 degrees
        self.joint_angles = np.clip(new_angles, -180, 180)
        print("New Joint Angles:", self.joint_angles)
        # Return the delta angles
        return delta_angles


    def dh_transform_matrix(self,a, d, alpha, theta):
    
    ## compute the standard Denavit-Hartenberg transformation matrix.
    ## source: wikipedia
        return np.array([
                [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
                [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
                [0, np.sin(alpha), np.cos(alpha), d],
                [0, 0, 0, 1]
            ])

    def forward_kinematics(self, theta_degrees):
        """
        Calculate the end-effector position using the provided joint angles (theta).
        """
        theta = np.radians(theta_degrees) # convert angles from degree to radians for cos and sin functions
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
            #a, d, alpha, theta_val = params
            #print(f"DH Parameters for joint {i}: a={a}, d={d}, alpha={alpha}, theta={theta_val}")

            T_i = self.dh_transform_matrix(*params)
            #print(f"Transformation Matrix T{i}:\n{T_i}\n")

            # Überprüfe die Form der Transformationsmatrix
            if T_i.shape != (4, 4):
                raise ValueError(f"Unexpected shape of transformation matrix T{i}: {T_i.shape}. Expected (4, 4)")

            T = np.dot(T, T_i)
        #print(f"Final Transformation Matrix T:\n{T}\n")  # Neu hinzugefügter Code

        # for params in dh_params:
        #     T = np.dot(T, self.dh_transform_matrix(*params))

        # extract position from the final transformation matrix
        position = T[:3, 3]
        return position

    # def tcp_position_to_grid_index(self, tcp_position):
    #     """Converts the TCP position to voxel grid indices."""
    #     # Implement conversion from TCP position to grid indices based on your environment's specifics
    #     # This is a placeholder function; you'll need to adjust it based on how your environment and TCP positions are defined
    #     #print("TCP Position:", tcp_position)
    #     x_idx = int(round((tcp_position[0] - self.x_range[0]) / self.resolution))
    #     y_idx = int(round((tcp_position[1] - self.y_range[0]) / self.resolution))
    #     z_idx = int(round((tcp_position[2] - self.z_range[0]) / self.resolution))
    #     print("TCP Position Indices:", x_idx, y_idx, z_idx)
    #     return x_idx, y_idx, z_idx
    
    def position_to_voxel_indices(self, point_in_voxel_space):
        # point_in_voxel_space already needs to be translated to voxel space
        x_idx = int(round(point_in_voxel_space[0] / self.resolution))
        y_idx = int(round(point_in_voxel_space[1] / self.resolution))
        z_idx = int(round(point_in_voxel_space[2] / self.resolution))
        return x_idx, y_idx, z_idx

    def embed_tcp_position(self, tcp_position):
        """Embeds the TCP position in a 3D grid of the same shape as the voxel space."""
        grid = np.zeros(self.voxel_space.shape, dtype=np.float32)
        # TCP position to grid index + set to 1
        #x_idx, y_idx, z_idx = self.tcp_position_to_grid_index(tcp_position)
        x_idx, y_idx, z_idx = self.position_to_voxel_indices(tcp_position)
        try:
            grid[x_idx, y_idx, z_idx] = 1
        except:
            print("TCP Position Indices out of bounds!")
        #print("TCP Position Grid:", grid)
        return grid


if __name__ == "__main__":
    env = RobotEnvironment()
    env.render(tcp_coords=env.tcp_position)
    
    #joint_angles = np.array([85.06, 24.65, 164.58, 179.33, -179.90, 0. ])
    #joint_angles= np.array([ 120., 35.06747416 , 163.77471266 , 180.  ,  -180., 0.   ])
    #joint_angles = np.array([ 89.99683147,  23.85781021, 164.55283017, 179.55921263 ,180., 180. ])
    #tcp_position = env.forward_kinematics(joint_angles)
    #print("TCP Position env:", tcp_position)
    #x,y,z= env.tcp_position_to_grid_index(tcp_position)
    #print("TCP Position Indices:", x, y, z)
    #state_size = np.prod(env.observation_space.shape)
    # print("action space: ", env.action_space) # action space:  MultiDiscrete([3 3 3 3 3 3])
    # print("obs space:", env.observation_space.shape)  #obs space: (2, 61, 61, 101)
    # action_size = env.action_space.shape[0] #.nvec.prod()   #Action size: (6,)  # tuple
    # print(f"State size: {state_size}, Action size: {action_size}") #State size: 751642

    # print("channels: ", env.observation_space.shape[0])  # channels: 2