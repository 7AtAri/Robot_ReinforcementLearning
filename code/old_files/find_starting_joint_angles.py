# find the starting joint angles 
# for the 6-joint robot to reach the starting position 
# of the helix in the voxel space

import numpy as np
from scipy.optimize import minimize

# DH parameters
dh_params = [
    (0, 0.15185, np.pi/2),
    (-0.24355, 0, 0),
    (-0.2132, 0, 0),
    (0, 0.13105, np.pi/2),
    (0, 0.08535, -np.pi/2),
    (0, 0.0921, 0)
]

# forward kinematics: calculate the TCP for specific joint angles
def forward_kinematics(theta):
    T = np.eye(4)
    for i, (a, d, alpha) in enumerate(dh_params):
        ct = np.cos(theta[i])
        st = np.sin(theta[i])
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        T_i = np.array([[ct, -st*ca,  st*sa, a*ct],
                        [st,  ct*ca, -ct*sa, a*st],
                        [0,      sa,     ca,    d],
                        [0,       0,      0,    1]])
        T = np.dot(T, T_i)
    return T[:3, 3]  # return x, y, z position

# objective function to minimize
def objective(theta):
    target_position = np.array([0.03, 0, 0])  # starting position of the helix in the voxel space
    position = forward_kinematics(theta)
    return np.linalg.norm(position - target_position)

# initial guess for theta
#theta_guess = np.zeros(6)
theta_guess= [120., 35.06747416 , 163.77471266 , 180.  ,  -180., 0.   ]
# solve for theta
result = minimize(objective, theta_guess, bounds=[(-np.pi, np.pi)] * 6)

if result.success:
    starting_angles = result.x
    print("joint angles (in degrees) TCP start pos:", np.degrees(starting_angles))
else:
    print("could not find a solution")



## 1st Output:
## [ 90.00106854   89.99929501  180.   62.13891154 -150.66767822  0.]    --->  [90, 90, 180, 62.14, -150.67 0]
## 2nd Output:
 # [  85.06175131   24.65057619  164.57837099  179.33239645 -179.90017193 0.  ]