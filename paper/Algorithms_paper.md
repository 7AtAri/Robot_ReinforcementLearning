# Algorithms tested by the authors

- Free Gait Generation with Reinforcement Learning (FGGRL) algorithm for a six-legged robot: This algorithm incorporates both free gait generation (FGG) and reinforcement learning (RL) to improve gait stability and adaptability, particularly focusing on challenges such as leg deficiencies.

## FGGRL-Algorithm components

- **Free Gait Generation (FGG)**: This component is responsible for generating stable free states for the robot's movement.
This ensures that gait is statically stable based on the "rule of neighborhood".
The "rule of neighborhood" guarantees that if a leg is lifted, both neighboring legs must be supporting.

- **Reinforcement Learning (RL)**: RL is utilized to learn the optimal transitions between states. This improves the gait stability over time. The reward to be maximised is tied to the stability margin of the robot's gait. It is also used for adapting to new conditions like speed changes or leg deficiencies (walking on 5-legs)

## Algorithm Choice

1. **Adaptability**: The combination of FGG and RL provides a good framework for the robot to adapt a gait in real-time and for various conditions, including leg deficiencies and speed changes. The adaptability is helpful for robots operating in unpredictable environments.

2. **Stability Focus**: The algorithm's emphasis on stability, both in terms of the reinforcement learning reward structure and the rule of neighborhood for gait generation, ensures that the robot maintains a stable gait under various conditions. This focus is essential for the practical deployment of legged robots on uneven terrain.

3. **Continuous Learning and Improvement**: Unlike some prior approaches that might rely heavily on pre-programmed gaits or offline learning, FGGRL enables continuous learning and improvement based on real-time feedback. This aspect makes it a strong choice for applications where the robot might encounter novel obstacles or terrain changes.

4. **Challenges in Real-time Application**: While the algorithm is designed to be adaptable and capable of continuous learning, the complexity of real-time reinforcement learning in robotics can pose significant challenges, especially regarding computational resources and the time required for learning stable gaits.

5. **Limited by Initial Gait Assumptions**: The effectiveness of the algorithm is partly dependent on the initial assumptions made during the free gait generation process, such as the rule of neighborhood. While this provides a stable starting point, it might limit the exploration of potentially more efficient or novel gait patterns not covered by the initial assumptions.
