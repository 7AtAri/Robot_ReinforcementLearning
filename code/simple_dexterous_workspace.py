import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Given DH parameters: (a, d, alpha)
dh_params = [
    (0, 0.15185, np.pi/2),
    (-0.24355, 0, 0),
    (-0.2132, 0, 0),
    (0, 0.13105, np.pi/2),
    (0, 0.08535, -np.pi/2),
    (0, 0.0921, 0)
]

def dh_transform(a, d, alpha, theta):
    """
    Calculate the Denavit-Hartenberg transformation matrix.
    """
    return np.array([
            [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
            [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])

# Simplified forward kinematics calculation
def calculate_position(dh_params, thetas):
    T = np.eye(4)  # Initialize transformation matrix to identity matrix
    for (a, d, alpha), theta in zip(dh_params, thetas):
        T_local = dh_transform(a, d, alpha, theta)
        T = np.dot(T, T_local)
    return T[:3, 3]

# Generate random sample joint angles within the range of -180° to 180°
num_samples = 1000
samples = np.random.uniform(low=-np.pi, high=np.pi, size=(num_samples, len(dh_params)))

# Calculate positions for each sample
positions = np.array([calculate_position(dh_params, sample) for sample in samples])


# Assuming 'positions' contains the X, Y, Z coordinates from the previous step

# Find min and max in each dimension
x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
z_min, z_max = positions[:, 2].min(), positions[:, 2].max()

# Output intervals in meters
print(f"X Interval: [{x_min:.2f}, {x_max:.2f}] meters")
print(f"Y Interval: [{y_min:.2f}, {y_max:.2f}] meters")
print(f"Z Interval: [{z_min:.2f}, {z_max:.2f}] meters")

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(positions[:,0], positions[:,1], positions[:,2], alpha=0.1)
ax.set_title('Approximate Dexterous Workspace')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
