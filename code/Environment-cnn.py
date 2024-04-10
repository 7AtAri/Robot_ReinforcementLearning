### ###############################################################################################################
### Reinforcement Learning - WS 2023/24 
### 6-joint robot arm
### Group: Dennis Huff, Philipp Bodemann, Ari Wahl 
### ###############################################################################################################

# https://gymnasium.farama.org/api/env/


import os
import shutil

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

        # combined observation space for voxel_space and TCP position  
        num_channels = 2  # # spatial observation space setup for a two-channel input:
        self.tcp_data_length = 3  # 3 orientation angle values
        # combine the two observation spaces with gym.tuple: spatial data and TCP data
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Box(low=-1, high=1, shape=(num_channels, self.x_size, self.y_size, self.z_size), dtype=np.float64), #  spacial data observation space defined by dimensions and possible voxel values (-1 to 1)
            gym.spaces.Box(low=-360, high=360, 
                                                shape=(self.tcp_data_length,), 
                                                dtype=np.float64) # observation space for the TCP data
        ))

        # init voxel space 
        self.voxel_space = np.full((self.x_size, self.y_size, self.z_size), -1)  # initialize all voxels with -1
        
        self.initial_joint_angles = np.array([0,0,0,0,0,0.0])  # initial joint angles
        self.initial_tcp_position, self.init_orientation = self.forward_kinematics(self.initial_joint_angles)  # initial end-effector position
        #print("Initial Robot TCP Position:", self.initial_tcp_position)
        #print("Orientierung (Roll, Pitch, Yaw):", self.init_orientation)

        self.joint_angles = self.initial_joint_angles  # set joint angles to initial joint angles
        
        # voxel space origin set to the initial TCP position:
      
        self.init_translation_matrix()
        # Populate the voxel space with a helix
        self.init_helix()

        self.tcp_position = self.translate_robot_to_voxel_space(self.initial_tcp_position)
        self.old_tcp_position = self.tcp_position
        #print("Initial Voxel TCP Position:", self.tcp_position)

        # # set the TCP position for the voxel space channel 2
        self.tcp_observation = self.embed_tcp_position(self.tcp_position)
        # stack to create a two-channel observation
        self.observation = np.stack([self.voxel_space, self.tcp_observation], axis=0)

        self.tcp_on_helix = self.is_on_helix(self.tcp_position)  # is the TCP is on the helix?
        #print("TCP on Helix:", self.tcp_on_helix)
        self.tcp_orientation= self.init_orientation
        self.reward = 0 # reward points
        self.terminated = False
        self.truncated = False
        self.out_of_voxel_space = False 

        # tcp orientation
        self.tolerance = 10 # 10 ° tolerance
        self.constant_orientation = (0, 0, 180)  # Roll-, pitch- und yaw in rad
        #self.last_orientation_deviation = 0  # Initialization of the variable for storing the previous orientation deviation
        #ori_hold = np.all(ori_diff <= self.tolerances[1]) or np.all(ori_diff >= (360-self.tolerances[1]))  
        
        # tcp pos tolerance
        self.tolerance_tcp_pos = 0.00142 # in stead of 1 mm tolerance diagonale vom Voxel

        # helixpoints
        self.helix_points = 0

        # closest distance
        self.closest_distance = None
        self.closest_point = None

        # counter for figures names 
        self.figure_count = 1

        # tcp_data
        self.tcp_data = np.asarray([self.init_orientation]) 

        # tuple of spatial and tcp data
        self.state = (self.observation, self.tcp_data)
      


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
            info (dict): A dictionary containing other information from the environment
        """
        # convert action to delta angles and apply them
        delta_angles = self.process_action(action)

        # Convert delta_angles to numpy array
        delta_angles = np.array(delta_angles)

        # Ensure delta_angles has the same dtype as self.joint_angles
        delta_angles = delta_angles.astype(self.joint_angles.dtype)

        #print("Joint Angles in step:", self.joint_angles)
        # update TCP position (based on the new joint angles - not on the delta angles) 
        new_tcp_position_in_robot_space, self.tcp_orientation = self.forward_kinematics(self.joint_angles)  # self.joint_angles are updated in process_action
        #print("new_TCP Position in robot space (step):", new_tcp_position_in_robot_space)
        #print("new Orientierung (Roll, Pitch, Yaw) in step:", tcp_orientation)
        self.old_tcp_position = self.tcp_position # save the old tcp position for the reward function
        self.tcp_position = self.translate_robot_to_voxel_space(new_tcp_position_in_robot_space)
        #print("New Voxel TCP Position in step:", self.tcp_position)

        # set the TCP position for the voxel space channel 2
        self.tcp_observation = self.embed_tcp_position(self.tcp_position)
        
        # stack to create a two-channel observation
        spatial_data = np.stack([self.voxel_space, self.tcp_observation], axis=0)

        # tcp_data
        self.tcp_data = np.asarray([self.tcp_orientation])

        # tuple of spatial and tcp data
        self.state = (spatial_data, self.tcp_data)
     
        # is TCP on the helix?
        self.tcp_on_helix = self.is_on_helix(self.tcp_position)

        # update the reward (based on the new state)
        self.reward = self.reward_function(self.tcp_on_helix)

        # eventually also return an info dictionary (for debugging)
        info = {
            'robot_state': self.joint_angles.tolist(),
            'tcp_position': self.tcp_position.tolist(), # current TCP position in voxel space
            'closest_point': self.closest_point.tolist(), # closest point on the helix
            'closest_distance': self.closest_distance.tolist(), # closest distance to the helix
            'current_orientation': self.tcp_orientation, # current orientation of the TCP
            'tcp_on_helix': self.tcp_on_helix # is the TCP on the helix?
        }

        # has to return: new observation (state), reward, terminated(bool), truncated(bool) info(dict)
        # return the new observation (state), reward, done flag
        return self.state, self.reward, self.terminated, self.truncated, info


    def init_helix(self):
        """
        Initialize a helix path in the voxel space.

        This function computes the coordinates of points along a helix path
        defined by the given radius, height per turn, and number of turns.

        Parameters:
            self

        Returns:
            None
        """

        # initialize helix
        r = self.radius  # radius 
        h = self.height_per_turn # height per turn 
        helix_resolution = 200
        t = np.linspace(0, self.turns, num=int(self.turns*helix_resolution) ) # parameter t from 0 to 2 for 2 complete turns

        offset = self.radius
        helix_x = r * np.cos(2 * np.pi * t + np.pi)  + offset  
        helix_y = r * np.sin(2 * np.pi * t + np.pi)  
        helix_z = h * t  

        # Initialize an empty list to store the helix points
        self.helix_points_list = []
        self.helix_points_list = [helix_x, helix_y, helix_z]

        # mark the voxels on the helix path:
        for i in range(len(helix_x)):
            x_idx = int(round((helix_x[i] - self.x_range[0]) / self.resolution))
            y_idx = int(round((helix_y[i] - self.y_range[0]) / self.resolution))
            z_idx = int(round((helix_z[i] - self.z_range[0]) / self.resolution))
            #self.helix_points_list.append([x_idx, y_idx, z_idx])
            if 0 <= x_idx < self.x_size and 0 <= y_idx < self.y_size and 0 <= z_idx < self.z_size:
                if i == len(helix_x) - 1:  # last helix point
                    self.voxel_space[x_idx, y_idx, z_idx] = 1
                else:
                    self.voxel_space[x_idx, y_idx, z_idx] = 0  # helix path
            else:
                print(f"Helix point out of bounds: {x_idx}, {y_idx}, {z_idx}")

        
        # Convert the list of indices to a numpy array and store it in self.helix_points
        self.helix_points = np.array(self.helix_points_list)

        # Print the helix points
        #print("Helix points:")
        #print(self.helix_points)

    def is_on_helix(self, tcp_coords):
        """
        Check if the TCP (Tool Center Point) coordinates lie on the helix path.

        This function converts the TCP coordinates to voxel indices within the voxel space
        and determines whether the TCP is on the helix path, at the target/end of the helix, 
        or outside the helix voxels.

        Parameters:
            tcp_coords (tuple): The TCP coordinates in the form of (x, y, z).

        Returns:
            bool: True if TCP is on the helix path or at the target, False otherwise.
        """
        # convert TCP coordinates to voxel indices. Therefore find the relative position of the TCP
        # within the bounds  `x_range`, `y_range`, and `z_range` 
        # scale this position to the resolution of the voxel grid:
        x_idx = int(round((tcp_coords[0] - self.x_range[0]) / self.resolution))
        y_idx = int(round((tcp_coords[1] - self.y_range[0]) / self.resolution))
        z_idx = int(round((tcp_coords[2] - self.z_range[0]) / self.resolution))

        print(f"TCP coords: {tcp_coords} -> Voxel indices: x:{x_idx}, y:{y_idx}, z:{z_idx}")  # debugging info

        # check if these indices are in the voxel space. If not, the TCP is outside the voxel space.
        if 0 <= x_idx < self.voxel_space.shape[0] and 0 <= y_idx < self.voxel_space.shape[1] and 0 <= z_idx < self.voxel_space.shape[2]:
            # get value of the voxel at the calculated indices. check if voxel
            # is on the helix path (0), the target/end of the helix (1), or outside the helix voxels (-1).
            voxel_value = self.voxel_space[x_idx, y_idx, z_idx]
            
            # if the TCP has reached the target (voxel-value = 1):
            if voxel_value == 1:
                print("TCP reached the target!")
                self.terminated = True
                self.out_of_voxel_space = False
                return True  # TCP is on the helix 
            
            # TCP is on a voxel of helix path but has not yet reached the end yet (voxel-value = 0):
            elif voxel_value == 0:
                print("TCP is on the helix path.")
                self.out_of_voxel_space = False
                return True # TCP is on the helix path
            
            elif voxel_value == -1:
                # Check if the distance to the helix is less than or equal to 0.001
                self.closest_point, self.closest_distance = self.find_closest_helix_point(tcp_coords, self.helix_points)
                #max_distance = np.max(closest_distance)
                if self.closest_distance <= self.tolerance_tcp_pos:
                    print("TCP is close to the helix.")
                    self.truncated = False
                    self.out_of_voxel_space = False
                    return True
                else:
                    print("TCP is outside the helix voxels.")
                    self.truncated = True
                    self.out_of_voxel_space = False
                    return False
            
        else:
            self.truncated = True
            print("TCP is outside the voxel space.")
            # otherwise the TCP is not on the helix path any more
            self.out_of_voxel_space = True
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
        print("__________________Resetting the environment...___________________")
        _ = seed  # acknowledging the seed parameter without using it to fit the gymnasium requirements
        self.voxel_space.fill(-1)

        self.initial_joint_angles = np.array([0,0,0,0,0,0.0])  # initial joint angles
        # self.initial_tcp_position = self.forward_kinematics(self.initial_joint_angles)  # initial end-effector position
        self.joint_angles = self.initial_joint_angles  # set joint angles to initial joint angles
        # voxel space origin set to the initial TCP position:
        #print("Initial TCP Position (Reset):", self.initial_tcp_position)
        self.init_translation_matrix()
        # Populate the voxel space with a helix
        self.init_helix()
        self.tcp_position = self.translate_robot_to_voxel_space(self.initial_tcp_position)

        #print("Voxel TCP Position (Reset):", self.tcp_position)
        # reset the joint angles and TCP position to the start of the helix
    
        self.tcp_observation = self.embed_tcp_position(self.tcp_position) # initial end-effector position
    
        # Stack to create a two-channel observation
        self.spatial_data= np.stack([self.voxel_space, self.tcp_observation], axis=0)

        # tcp_data
        self.tcp_data = np.asarray([self.init_orientation])

        # tuple of spatial and tcp data
        self.state = (self.spatial_data, self.tcp_data)

        # reset the reward and Flags
        self.tcp_on_helix = self.is_on_helix(self.tcp_position)
        self.reward = 0
        self.terminated= False
        self.truncated = False

        # counter for figures names
        self.figure_count = 1

        # eventually also return an info dictionary (for debugging)
        info = {
            'robot_state': self.initial_joint_angles.tolist(),
            'tcp_position': self.tcp_position.tolist()
        }

        return self.state, info #  self.joint_angles  # also return the joint angles?


    def reward_function(self, tcp_on_helix):
        """Calculate the reward based on the current state of the environment."""
        self.reward = 0
        #closest_point, closest_distance = self.find_closest_helix_point(, self.helix_points)
        _, orientation_deviation, _ = self.objective_function_with_orientation(self.joint_angles, self.constant_orientation)  # Roll, Pitch, Yaw in Grad
       
        # initialize reward, terminated, and truncated flags
        if tcp_on_helix and self.tcp_position[2] >= self.old_tcp_position[2]:
            self.reward += 10
            self.truncated = False
        if self.terminated:
            self.reward += 1000 # extra reward for reaching the target
            
        if self.truncated:
             # terminate the episode if the tcp is not on the helix any more
            self.reward -=1

        # Adjust reward based on orientation deviation
        orientation_reward = 0
        max_deviation = np.max(orientation_deviation)
        print("max_deviation (in reward)", np.round(max_deviation,2))
        if max_deviation <= self.tolerance:
            orientation_reward = 10
        else:
            orientation_reward = 0

        if self.out_of_voxel_space:
            self.reward -= 10
        # Add orientation reward to total reward
        self.reward += orientation_reward

        return self.reward
    

    def render(self):
        """

        This function visualizes the voxel space with the helix path and highlights the TCP position
        if provided and valid.
            
        Returns:
            None
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*np.where(self.voxel_space == 1), c='r', s=40, alpha=1)  # helix end points
        ax.scatter(*np.where(self.voxel_space == 0), c='b', s=40, alpha=0.4)  # helix path points
        
        # if TCP coordinates are provided and valid, then visualize TCP position
        if self.tcp_position is not None:
            #print(f"Is TCP on Helix Path: {is_on_path}")

            # convert real-world coordinates to indices for visualization
            x_idx = (self.tcp_position[0] - self.x_range[0]) / self.resolution
            y_idx = (self.tcp_position[1] - self.y_range[0]) / self.resolution
            z_idx = (self.tcp_position[2] - self.z_range[0]) / self.resolution
            
            # highlight TCP position
        ax.scatter([x_idx], [y_idx], [z_idx], c='orange', s=100, alpha= 1, label='TCP Position')
        # Erstellen Sie den Pfeil für die Orientierung
        #ax.quiver(x_idx, y_idx, z_idx, self.tcp_orientation[0], self.tcp_orientation[1], self.tcp_orientation[2], color='black', length=15, normalize=True, arrow_length_ratio=0.2, linewidth=1)
        ax.quiver(x_idx, y_idx, z_idx, self.constant_orientation[0], self.constant_orientation[1], self.constant_orientation[2], color='black', length=15, normalize=True, arrow_length_ratio=0.2, linewidth=1)
        # Set axis limits to start from 0
        #ax.set_xlim(0, self.x_size)
        #ax.set_ylim(0, self.y_size)
        #ax.set_zlim(0, self.z_size)
        ax.set_xlabel('X Index')
        ax.set_ylabel('Y Index')
        ax.set_zlabel('Z Index')
        ax.set_title('3D Plot of the Voxel Space')
        #plt.legend()
        #plt.show()

        # Create directory if not exists
        if not os.path.exists('ParamCombi1'):
            os.makedirs('ParamCombi1')
        # Create directory if not exists
        if not os.path.exists('ParamCombi2'):
            os.makedirs('ParamCombi2')
        
        # to check if a new episode has started
        new_episode = False 
        
        # check if one of the folders contains the MSE file:
        # Specify the folder path and the filename
        folder_path1 = 'ParamCombi1'
        folder_path2 = 'ParamCombi2'
        filename = 'MSE.png'

        # Construct the full path to the file
        file_path1 = os.path.join(folder_path1, filename)
        file_path2 = os.path.join(folder_path2, filename)

        if os.listdir('ParamCombi1') == [] and os.listdir('ParamCombi2') == []:
            new_episode = True

        # check if the MSE file exists in folder (this means that this folders episode has ended)
        if os.path.exists(file_path1):
            if os.path.exists(file_path2):
                new_episode = True
        elif os.path.exists(file_path2) and os.listdir('ParamCombi1') == []:
                new_episode = True
        elif os.path.exists(file_path1) and os.listdir('ParamCombi2') == []:
                new_episode = True


        if new_episode:
            #print("New Episode")
            # check which folder contains more elements and delete the one with less elements and save in the folder which has less files
            num_files_in_ParamCombi1 = len(os.listdir("ParamCombi1"))
            num_files_in_ParamCombi2 = len(os.listdir("ParamCombi2"))    

            if num_files_in_ParamCombi1 >= num_files_in_ParamCombi2:
                    self.current_directory = 2
                    #print("Current Directory: 2")
                    shutil.rmtree('ParamCombi2')
                    os.makedirs('ParamCombi2')
                    plt.savefig(os.path.join('ParamCombi2', f'step_{self.figure_count}.png'))
            else:
                    self.current_directory = 1
                    #print("Current Directory: 1")
                    shutil.rmtree('ParamCombi1')
                    os.makedirs('ParamCombi1')
                    plt.savefig(os.path.join('ParamCombi1', f'step_{self.figure_count}.png'))
        else:
            if self.current_directory == 1:
                # Save the figure in folder 1
                plt.savefig(os.path.join('ParamCombi1', f'step_{self.figure_count}.png'))
            else:
                # Save the figure in folder 2
                plt.savefig(os.path.join('ParamCombi2', f'step_{self.figure_count}.png'))          

        plt.close() # close the figure  
        
        self.figure_count += 1 # increment the figure count


    def process_action(self, action):
        """
        Process the action provided to generate new joint angles.

        This function calculates the delta angles for each action and updates the joint angles accordingly.
        The delta angles are calculated based on the action values provided, and the joint angles are limited
        within the range of -180 to 180 degrees.

        Parameters:
            action: The action values to be processed.

        Returns:
            np.ndarray: The delta angles resulting from the action processing.
        """
        # Check if action is iterable
        if isinstance(action, (list, tuple)):
        # If yes, calculate the delta angles for each action
            delta_angles = np.array([(a - 1) * 0.1 for a in action])
            #print("Delta Angles (process action):", delta_angles)
        else:
        # Otherwise, there is only one action, so calculate the delta angle directly
            delta_angles = np.array([(action - 1) * 0.1])
            
        #print("joint_angles (process action):", self.joint_angles)
        new_angles = self.joint_angles + delta_angles

        # Limit the new joint angles within the range of -180 to 180 degrees
        self.joint_angles = np.clip(new_angles, -180, 180)
        #print("New Joint Angles (process action):", self.joint_angles)
        # Return the delta angles
        return delta_angles

    def init_translation_matrix(self):
        """
        Initialize the translation matrix based on the initial TCP position.

        This function computes the translation vector as the negative of the initial TCP position,
        and constructs the translation matrix accordingly.

        Returns:
            None
        """
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
        # update voxel grid as needed
        return x_idx, y_idx, z_idx

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
        Calculate the end-effector position and orientation using the provided joint angles.

        This function computes the end-effector position and orientation based on the Denavit-Hartenberg (DH) parameters
        and the given joint angles.

        Parameters:
            theta_degrees: Joint angles in degrees.

        Returns:
            tuple: A tuple containing the end-effector position (x, y, z) and orientation angles (alpha, beta, gamma).
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
        # extract rotation matrix from devanit-hartenberg matrix
        rotation_matrix = T[:3, :3]
        # # https://de.wikipedia.org/wiki/Roll-Nick-Gier-Winkel
        # calculate beta
        beta = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2))

        # calculate alpha
        alpha = np.arctan2(rotation_matrix[1, 0] / np.cos(beta), rotation_matrix[0, 0] / np.cos(beta))

        # calculate gamma
        gamma = np.arctan2(rotation_matrix[2, 1] / np.cos(beta), rotation_matrix[2, 2] / np.cos(beta))

        # normalize angles
        alpha, beta, gamma = alpha % (2*np.pi), beta % (2*np.pi), gamma % (2*np.pi)

        # convert angles to degrees
        alpha,beta,gamma = np.rad2deg([alpha,beta,gamma])

        self.tcp_orientation = (alpha, beta, gamma)
        #print("position_tcp: ", position)
        return position, (alpha, beta, gamma)


    def objective_function_with_orientation(self, theta, constant_orientation): # closes_target_pos
        """
        Calculate the combined positional and orientational error for the robot end-effector.
        """
        # Calculate the current position and orientation from forward kinematics
        current_position, current_orientation = self.forward_kinematics(theta) # joint angle
        current_position = self.translate_robot_to_voxel_space(current_position)
        # get closest point (closest_target_pos)xxx
        #print("current_tcp_pos_in_voxel_space (objective func):", current_position)
        closest_helix_point, closest_distance = self.find_closest_helix_point(current_position, self.helix_points)

        # Format the orientation to two decimals
        formatted_orientation = [f'{num:.2f}' for num in current_orientation]
        print(formatted_orientation)
        # Calculate the positional error
        position_error = np.linalg.norm(np.array(current_position) - np.array(closest_helix_point))
        
        # Convert orientation tuples to numpy arrays
        current_orientation = np.array(current_orientation)
        constant_orientation = np.array(constant_orientation)
        
        # deviation of current and constant orientation
        orientation_errors = np.abs(current_orientation - constant_orientation)
        # Combine errors, possibly with weighting factors if needed
        #total_error = position_error + orientation_error

        return position_error, orientation_errors, closest_distance
    


    def find_closest_helix_point(self, current_tcp_position, helix_points):
        """
        Find the closest point on the helix to the current TCP position.
        """
        # calculate distance between every helix point and every current tcp position
        differences = helix_points - current_tcp_position.reshape(3, 1)
        distances = np.linalg.norm(differences, axis=0)

        # find index with smallest distance
        closest_index = np.argmin(distances)

        # Return of the point on the helix closest to the current TCP position and the corresponding distance
        closest_point = helix_points[:, closest_index]
        closest_distance = distances[closest_index]

        #print("closest point ", closest_point)
        #print("closest distance ", np.round(closest_distance,4)) # double
        #x_idx = int(round(closest_point[0] / self.resolution))
        x_idx = (closest_point[0] - self.x_range[0]) / self.resolution
        y_idx = (closest_point[1] - self.y_range[0]) / self.resolution
        z_idx = (closest_point[2] - self.z_range[0]) / self.resolution
        test = [ x_idx,  y_idx,  z_idx]
        #print("closest_point invoxel_space", test)
        self.closest_distance = closest_distance
        self.closest_point = closest_point

        return closest_point, closest_distance



    def position_to_voxel_indices(self, point_in_voxel_space):
        """
        This function converts the coordinates of a point in voxel space to voxel indices,
        considering the resolution of the voxel grid.

        Parameters:
            point_in_voxel_space: The coordinates of the point in voxel space.

        Returns:
            tuple: A tuple containing the voxel indices (x_idx, y_idx, z_idx) of the given point.
        """
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
    env.render()
    
