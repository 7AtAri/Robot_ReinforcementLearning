# best helix position in the robots reach

Strategically position the helix so that the robot's TCP can follow its trajectory within the robot's range and considering its joint angle constraints from -180° to 180°:

1) Helix Dimensions:  relatively small and manageable
2) Robot Reach and Workspace:

- total reach of the robot and its workspace geometry should surround the helix's trajectory
- position helix's base point within the robot's workspace, centered as much as possible to maximize accessibility.

3) Joint Constraints:

- joints can rotate from -180° to 180° --> wide range of motions.
- helix must be positioned so that the entire path remains within the robot's dexterous workspace where it can maintain good posture (avoiding singularities and joint limits)

## Task Space of a robot

- [task space](https://mecharithm.com/learning/lesson/task-space-and-workspace-for-robots-102)

- [reachable vs dexterous workspace of a robot](https://firstyearengineer.com/intro-to-robotics/chapter-2-robot-manipulators/classifications-robot-workspaces/)

## Positioning Strategy for the Helix

Given the robot's configuration at the origin (0, 0, 0) with all joint angles set to 0 degrees, the initial position and orientation of the TCP in the robot's base frame need to be determined. This is the result of the forward kinematics with all joints at 0 degrees. Then, we need to adjust the position of the helix start point relative to this TCP position.

## Steps to Position the Helix

- Calculate TCP's Initial Position: Use forward kinematics with all joint angles set to 0 degrees to find the TCP's initial position in the robot's base frame.
- Determine Helix Start Position: Based on the helix parameters and the desired start position in the robot's workspace, determine where the start of the helix should be in the robot's base frame to ensure it's reachable by the TCP at the initial configuration.
- Calculate Translation: The translation vector is the difference between the TCP's initial position (step 1) and the desired start position of the helix (step 2). This vector tells you how to translate the helix so its start is at the TCP's initial position.
- Define voxel space around the helix and fill it with the helix, so that the helix starts at the origin of the voxel space
- Use the Translation matrix to translate between the robot's coordinate system and the voxel-space origin, that is set at the TCP for all joints = 0.


