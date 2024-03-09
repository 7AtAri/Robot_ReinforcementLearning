# Helix

To define a 3D trajectory mathematically one needs to use the equation of a helix with the given radius of 3 cm and 2 turns. The trajectory is defined in a 3D coordinate system (x, y, z):

- The x- and y-coordinates describe the circular position in the horizontal plane.
- The z-coordinate represents the vertical position. It increases (linearly?) with the helix going up.

mathematical representation of a helix:

```math
\vec{x}(t) = \begin{pmatrix}
r \cdot \cos(2\pi t) \\
r \cdot \sin(2\pi t) \\
h \cdot t + c
\end{pmatrix}
```

We choose, that each turn of the helix raises h=0.05 meters (5 cm). This leads to a total ascent of 0.1 meters for 2 turns. We assume **c = 0** for simplicity and define the trajectory:

parameters:

- \(r\): Radius of the helix (r = 3cm)
- \(h\): The height per turn of the helix (h = 5cm)
- \(c\): A constant, representing a height offset (c = 0)
- \(t\): The parameter \(t\) ranges from 0 to 2 for 2 complete turns

```math
\vec{x}(t) = \begin{pmatrix}
0.03 \cdot \cos(2\pi \cdot t) \\
0.03 \cdot \sin(2\pi \cdot t) \\
0.05 \cdot t
\end{pmatrix}
```
