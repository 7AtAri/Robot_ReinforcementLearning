# Reinforcement Learning for Robotics


## Paper: 6-legged Robot

- [x] read the [paper](https://github.com/7AtAri/Robot_ReinforcementLearning/blob/main/2023WiSe_2PZR_RL_05.pdf)
- [x] take [notes](https://docs.google.com/document/d/1rnW2lWlUQMY4ahw68WoXD42BBrlzWxlGkcxe9iR7dCI/edit)
      
Presentation of the paper (20 min):

- [x] make [slideshow](https://docs.google.com/presentation/d/1c81UuuYcv-xGZ58Bya9-mc7IEekxRP4rj4RAE2KFHCg/edit#slide=id.p)
- [ ] 20 min -> 20 slides / 8 Questions = 2,5 slides per topic -> 5 slides for each

Questions:

- [x] What is the problem? Describe the environment using mathematics (Ari)
- [x] How has the problem been solved so far? What are the drawbacks of that solution? (Philipp)
- [x] Why can RL solve the problem in a better manner? (Dennis)
- [x] What are the metrics (figures of merit)? When is a solution good enough for that problem? (Philipp)
- [x] Which Algorithms are the authors testing? Are they a good choice? (Dennis)
- [x] Do they achieve a better solution compared to the previous solutions? (Ari)
- [x] What could be changed / improved (in our opinion)?
- [x] A few simulations that confirm our proposals for improvement

Presentation of the project (20 min):

- [ ] make [slideshow](https://docs.google.com/presentation/d/1K-Z_9DINiN5YOrNhcSbybdJc9_H6uGj8DYdVeNjimRg/edit?usp=sharing)


## Project: 6-joint Robot Arm

- [x] read the [task](https://github.com/7AtAri/Robot_ReinforcementLearning/blob/main/2023%20WiSe_2PZR_Coding_Task_05.pdf)
- [x] [notes on the task](https://docs.google.com/document/d/1-oN-ch47fVDCPkOF1WOgRCZygJi9Pc00SBXlcemRMhU/edit?usp=sharing) 

- [x]	Define the 3D-trajectory mathematically.
- [x]	Define the Observation Space in the form of the voxel where the trajectory will be.
- [x] Code the Environment with the given direct kinematics and the constraints of the join angles and the chosen voxel.
      Define a suitable Reward Function for the Agent,
      as well as the strategy how to deal with occasions where the TCP leaves the trajectory and/or the voxel.
- [ ]	Code the defined algorithm to come to an optimal Policy.
- [ ]	Verify the performance vs. time of the Learning process showing that your Agent is in fact learning
      and the decisions are becoming “better” after each training.
- [ ] Once your Agent learned the optimal Policy, test the Policy
      showing that the robot’s TCP is indeed following the trajectory.
      Compute the mean square error (MSE) from the ideal trajectory and what the Agent is doing in reality.
- [ ]	Propose new ideas how to reduce the MSE …
      you don’t need to code them, but your ideas must be plausible and realistic: Substantiate them! 
- [ ]	Make sure that your code has some tools to allow for easy verification of the performance of your Agent.
