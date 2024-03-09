# Tool Center Point (TCP)

Calculating the Tool Center Point (TCP) position (x, y, z) and its orientation (α, β, γ) in a 3D space from the direct kinematics of a robot with six joints involves using the robot's joint angles $\theta_1$ to $\theta_6$ to compute the overall transformation matrix from the base of the robot to the TCP.

Steps:

1) Use Denavit-Hartenberg (D-H) convention to get the transformation matrices for each joint.

2) Multiply these matrices to get the final position and orientation of the TCP.

## 1: Compute Individual Transformation Matrices

For each joint, calculate its transformation matrix using its D-H parameters $a_i, d_i,\alpha_i$ and $\theta_i$. The transformation matrix for a joint **i** is given by:

```math
T_i = \begin{bmatrix}
\cos(\theta_i) & -\sin(\theta_i)\cos(\alpha_i) & \sin(\theta_i)\sin(\alpha_i) & a_i\cos(\theta_i) \\
\sin(\theta_i) & \cos(\theta_i)\cos(\alpha_i) & -\cos(\theta_i)\sin(\alpha_i) & a_i\sin(\theta_i) \\
0 & \sin(\alpha_i) & \cos(\alpha_i) & d_i \\
0 & 0 & 0 & 1
\end{bmatrix}
```

## 2: Multiply Transformation Matrices

Multiply all individual transformation matrices in sequence from the base to the TCP to get the overall transformation matrix $T_{TCP}$:

```math
T_{TCP} = T_1 \cdot T_2 \cdot T_3 \cdot T_4 \cdot T_5 \cdot T_6
```

## 3: Extract Position and Orientation

- **Position (x, y, z):** The position of the TCP in 3D space is given by the translation components of the final transformation matrix $T_{TCP}$, specifically, the elements in the last column of the first three rows $(T_{13}), (T_{23}), (T_{33})$.

- **Orientation (α, β, γ):** The orientation of the TCP can be extracted from the rotation part of $T_{TCP}$ and is often represented in terms of Euler angles or Roll-Pitch-Yaw angles. These angles are derived from the rotation matrix R (the top-left 3x3 submatrix of $T_{TCP}$)as follows:

  - **Roll (α)**: The rotation about the x-axis can be calculated as $\alpha = \text{atan2}(R_{32}, R_{33})$
  - **Pitch (β)**: The rotation about the y-axis can be calculated as $\beta = \text{atan2}(-R_{31}, \sqrt{R_{32}^2 + R_{33}^2})$
  - **Yaw (γ)**: The rotation about the z-axis can be calculated as $\gamma = \text{atan2}(R_{21}, R_{11})$

### Starting Point of the TCP

The starting position of the TCP corresponds to the value of $\vec{x}(t)$ at t=0 , which represents the beginning of the trajectory:

```math
\vec{x}(0) = \begin{pmatrix}
0.03 \cdot \cos(2\pi \cdot 0) \\
0.03 \cdot \sin(2\pi \cdot 0) \\
0.05 \cdot 0
\end{pmatrix} = \begin{pmatrix}
0.03 \\
0 \\
0
\end{pmatrix}
```
