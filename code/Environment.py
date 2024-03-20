import numpy as np
from math import cos, sin, pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class RobotEnvironment:
    def __init__(self):
        self.radius = 0.03  # Radius of the helix in meters
        self.height_per_turn = 0.05  # Height per turn in meters
        self.turns = 2  # Number of turns
        self.start_position = np.array([self.radius, 0, 0])  # Starting position of the TCP
        self.reward = 0 # reward points
        self.reached_target = False
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
    
    def reward_function(self,tcp_on_helix):
        if tcp_on_helix:
            self.reward += 1
        else:
            self.reward -=1 
           
    
    def is_on_helix(self,tcp_coords, voxel_space, resolution, x_range, y_range, z_range):
        # Convert TCP coordinates to voxel indices
        x_idx = int(round((tcp_coords[0] - x_range[0]) / resolution))
        y_idx = int(round((tcp_coords[1] - y_range[0]) / resolution))
        z_idx = int(round((tcp_coords[2] - z_range[0]) / resolution))

        # Check if the TCP is within the bounds of the voxel space
        if (x_idx < 0 or x_idx >= voxel_space.shape[0] or
            y_idx < 0 or y_idx >= voxel_space.shape[1] or
            z_idx < 0 or z_idx >= voxel_space.shape[2]):
            return False

        # Check the value of the voxel to determine if the TCP is on the helix
        if voxel_space[x_idx, y_idx, z_idx] == 1 or voxel_space[x_idx, y_idx, z_idx] == 1:
            if voxel_space[x_idx, y_idx, z_idx] == 1:
                self.reached_target = True
            return True
        else:
            return False




# Object of RobotEnvironment
env = RobotEnvironment()

# 1.) Define the observation space:

## Define the voxel space dimensions

x_range = (-0.03, 0.03)
y_range = (-0.03, 0.03)
z_range = (0, 0.1)

## Assuming a resolution of 1 mm (0.001 m) for the voxel space

resolution = 0.001

## Calculate the size of the voxel array for each dimension

x_size = int((x_range[1] - x_range[0]) / resolution) + 1  # +1 to ensure start and end of range are included because of the 0 voxel
y_size = int((y_range[1] - y_range[0]) / resolution) + 1
z_size = int((z_range[1] - z_range[0]) / resolution) + 1

## Create the numpy array for the voxel space

#voxel_space = np.zeros((x_size, y_size, z_size))
voxel_space = np.full((x_size, y_size, z_size), -1)  # Initialize all voxels with -1

print(voxel_space.shape)

# init Helix
r = 0.03  # Radius Helix
h = 0.05  # Höhe pro Umdrehung der Helix
t = np.linspace(0, 2, 100)  # Parameter t von 0 bis 2 für 2 komplette Umdrehungen
helix_x = r * np.cos(2 * np.pi * t)
helix_y = r * np.sin(2 * np.pi * t)
helix_z = h * t
#print("helix_x: ",helix_x)

# Markierung der Voxel auf der Helix als Rennstrecke
for i in range(len(helix_x)): # alle haben die selbe Länge...
    x_idx = int(round((helix_x[i] - x_range[0]) / resolution)) # umrechnung in index 
    #print(x_idx)
    y_idx = int(round((helix_y[i] - y_range[0]) / resolution))
    z_idx = int(round((helix_z[i] - z_range[0]) / resolution))
    if i == len(helix_x) - 1:  # Letzter Helixpunkt
        voxel_space[x_idx, y_idx, z_idx] = 1
    else:
        voxel_space[x_idx, y_idx, z_idx] = 0  # Helix

print("Voxelshape: ",voxel_space.shape)
print(voxel_space)

###############

# 3D-Plot Helix Voxels
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*np.where(voxel_space == 1), c='r', s=50, alpha=1)
ax.scatter(*np.where(voxel_space == 0), c='b', s=50, alpha=1)  # Leichter Druck für Voxel außerhalb der Helix
#ax.scatter(*np.where(voxel_space == -1), c='y', s=1, alpha=0.1)  # Leichter Druck für Voxel außerhalb der Helix

#ax.scatter(*tcp_coords_on_voxel, c='g', s=100, label='TCP Position')  # TCP Position

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D-Plot des Voxel-Raums')

plt.show()

# Test function with TCP position on voxel with value 0
tcp_coords_on_helix = (0.015, -0.02, 0.045)
result_0 = env.is_on_helix(tcp_coords_on_helix, voxel_space, resolution, x_range, y_range, z_range)
print("TCP position on voxel with value 0:", result_0)

# Test function with TCP position on voxel with value -1
tcp_coords_neg1 = (0.003, 0, 0.005)
result_neg1 = env.is_on_helix(tcp_coords_neg1, voxel_space, resolution, x_range, y_range, z_range)
print("TCP position on voxel with value -1:", result_neg1)