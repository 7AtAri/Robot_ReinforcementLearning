# Reinforcement Learning for Robotics

## Project: 6-joint Robot Arm

- [x] presentation [slides](https://docs.google.com/presentation/d/1K-Z_9DINiN5YOrNhcSbybdJc9_H6uGj8DYdVeNjimRg/edit?usp=sharing)
- [x] read the [task](https://github.com/7AtAri/Robot_ReinforcementLearning/blob/main/2023%20WiSe_2PZR_Coding_Task_05.pdf)
- [x] [notes on the task](https://docs.google.com/document/d/1-oN-ch47fVDCPkOF1WOgRCZygJi9Pc00SBXlcemRMhU/edit?usp=sharing) 

- [x]	3D-trajectory mathematically defined
- [x]	construct Observation Space in the form of the voxel and include the trajectory (helix)
- [x] Environment with the given direct kinematics and constraints of the join angles and the chosen voxel space.
      Suitable Reward Function for the Agent
      Strategy of how to deal with occasions where the TCP leaves the trajectory and/or the voxel.
- [x]	Coding the Deep Q-Learning algorithm to approach an optimal Policy.
- [x]	Performance vs. time of the Learning process
      --> read the log files, episodes get longer with time
- [x] Test the Policy (exploiting) showing that the robot’s TCP is indeed following the trajectory.
      Computing the mean square error (MSE) from the ideal trajectory and what the Agent is doing in reality.
      Plotting the MSE per episode after each Training run
- [ ]	new ideas how to reduce the MSE …
      must be plausible and realistic: Substantiate them! 
- [x]	Loggings and renderings for easy verification of the performance of our Agent.

### Presentation of the project (30 min):
upload your code + environment two days before the appointment

- [x] make [slideshow](https://docs.google.com/presentation/d/1K-Z_9DINiN5YOrNhcSbybdJc9_H6uGj8DYdVeNjimRg/edit?usp=sharing)

Questions:

- [x] Prepare a short summary of the problem and the way to solve it
- [x] Create a few test scenarios upfront to test the performance of your code
- [x] Document your code
- [x] Show the behaviour of the Environment (--> render())
- [x] Show that the learned policy performs well (more or less)
- [x] Be prepared to make a few changes in your code and re-run it showing the new performance!

Dependencies:
- pip install numpy
- pip install matplotlib
- pip install gymnasium
- pip install torch
- pip install scikit-learn


## Paper: 6-legged Robot

- [x] read the [paper](https://github.com/7AtAri/Robot_ReinforcementLearning/blob/main/2023WiSe_2PZR_RL_05.pdf)
- [x] take [notes](https://docs.google.com/document/d/1rnW2lWlUQMY4ahw68WoXD42BBrlzWxlGkcxe9iR7dCI/edit)
      
### Presentation of the paper (20 min):

- [x] make [slideshow](https://docs.google.com/presentation/d/1c81UuuYcv-xGZ58Bya9-mc7IEekxRP4rj4RAE2KFHCg/edit#slide=id.p)
- [x] 20 min -> 20 slides / 8 Questions = 2,5 slides per topic -> 5 slides for each

Questions:

- [x] What is the problem? Describe the environment using mathematics (Ari)
- [x] How has the problem been solved so far? What are the drawbacks of that solution? (Philipp)
- [x] Why can RL solve the problem in a better manner? (Dennis)
- [x] What are the metrics (figures of merit)? When is a solution good enough for that problem? (Philipp)
- [x] Which Algorithms are the authors testing? Are they a good choice? (Dennis)
- [x] Do they achieve a better solution compared to the previous solutions? (Ari)
- [x] What could be changed / improved (in our opinion)?
- [x] A few simulations that confirm our proposals for improvement

