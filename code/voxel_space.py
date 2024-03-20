### 
### Reinforcement Learning - WS 2023/24 
### 6-joint robot arm
### Group: Dennis Huff, Philip, Ari Wahl 
###


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

# Erstellen der Helix
r = 0.03  # Radius der Helix
h = 0.05  # Höhe pro Umdrehung der Helix
t = np.linspace(0, 2, 100)  # Parameter t von 0 bis 2 für 2 komplette Umdrehungen
helix_x = r * np.cos(2 * np.pi * t)
helix_y = r * np.sin(2 * np.pi * t)
helix_z = h * t

# Markierung der Voxel auf der Helix als Rennstrecke
for i in range(len(helix_x)):
    x_idx = int(round((helix_x[i] - x_range[0]) / resolution))
    y_idx = int(round((helix_y[i] - y_range[0]) / resolution))
    z_idx = int(round((helix_z[i] - z_range[0]) / resolution))
    voxel_space[x_idx, y_idx, z_idx] = 1  # Rennstrecke

print(voxel_space)

###############

# 3D-Plot Helix Voxels
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*np.where(voxel_space == 1), c='r', s=50, alpha=1)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D-Plot des Voxel-Raums')

plt.show()

##############



## 3D-Plot des Voxel-Raums
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(*np.where(voxel_space == 0), c='b', s=5, alpha=0.1)  # Leichter Druck für Voxel außerhalb der Helix
#ax.scatter(*np.where(voxel_space == 1), c='r', s=50, alpha=1)    # Starker Druck für Voxel auf der Helix
#
#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('Z')
#ax.set_title('3D-Gitter - Rennstrecke mit Helix')
#
#plt.show()