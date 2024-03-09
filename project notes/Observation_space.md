# Observation Space

- x Dimension: [−0.03, 0.03] m
- y Dimension: [−0.03, 0.03] m
- z Dimension: [0, 0.1] m

The voxel space can be described by a 3-dim np.array.

**starting code snippet for observation space:**

import numpy as np

## Define the voxel space dimensions

x_range = (-0.03, 0.03)
y_range = (-0.03, 0.03)
z_range = (0, 0.1)

## Assuming a resolution of 1 mm (0.001 m) for the voxel space

resolution = 0.001

## Calculate the size of the voxel array for each dimension

x_size = int((x_range[1] - x_range[0]) / resolution) + 1  # +1 to ensure start and end of range are included
y_size = int((y_range[1] - y_range[0]) / resolution) + 1
z_size = int((z_range[1] - z_range[0]) / resolution) + 1

## Create the numpy array for the voxel space

voxel_space = np.zeros((x_size, y_size, z_size))

voxel_space.shape
