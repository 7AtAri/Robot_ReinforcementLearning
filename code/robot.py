### 
### Reinforcement Learning - WS 2023/24 
### 6-joint robot arm
### Group: Dennis Huff, Philip, Ari Wahl 
###


import numpy as np

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

voxel_space = np.zeros((x_size, y_size, z_size))

print(voxel_space.shape)

