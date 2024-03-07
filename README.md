# Reinforcement Learning for Robotics


## Paper: 6-legged Robot

- [ ] read the paper
      
Presentation of the paper:

- [ ] What is the problem? Describe the environment using mathematics
- [ ] How has the problem be solved so far? What are the drawbacks of that solution?
- [ ] Why can RL solve the problem in a better manner?
- [ ] What are the metrics? When is a solution good enough for that problem?
- [ ] Which Algorithms are the authors testing? Are they a good choice?
- [ ] Do they achieve a better solution compared to the previous solutions?
- [ ] What could be changed / improved (in our opinion)?
- [ ] A few simulations that confirm our proposals for improvement


## Project: 6-joint Robot Arm

- [ ]	Define the 3D-trajectory mathematically.
- [ ]	Define the Observation Space in the form of the voxel where the trajectory will be.
- [ ] Code the Environment with the given direct kinematics and the constraints of the join angles and the chosen voxel.
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
