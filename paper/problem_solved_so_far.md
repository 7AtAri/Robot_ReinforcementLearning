Before the solution presented in the paper, the problem of generating free gaits for a six-legged robot was approached using different methodologies, as highlighted in the literature review section of the paper. Here are the key points regarding how the problem has been addressed in the past and the drawbacks of those solutions:

### Previous Approaches to Free Gait Generation

1. **Reinforcement Learning (RL) Applications**: RL has been widely applied to walking machines to improve various robot behaviors. These applications primarily focused on developing suitable protraction and retraction movements with off-line learning or real-time reinforcement learning to develop elementary movements for a single leg and then extend these to more complex gaits like the tripod gait for six-legged robots.

2. **Free Gait Generation with Off-line Learning or Programming**: There were attempts at free gait generation where off-line learning or pre-programming was used. These approaches involved developing optimal path and gait generation simultaneously, performing free gait generation by using graph search among predefined states, or proposing reactive free gait generation that modifies the gait based on robot-environment interactions. Such methods guaranteed stability at every step but were limited in their adaptability and continuous improvement capabilities.

### Drawbacks of Previous Solutions

1. **Limited Adaptability and Improvement**: One of the main drawbacks of previous solutions, especially those involving off-line learning or programming in advance, was their limited adaptability and scope for continuous improvement. While these methods provided a structure for real-time modification of gaits based on the robot's interaction with the environment, the modification was restricted to the capabilities of the off-line developed structures.

2. **Lack of Continuous Development Structure**: The preplanned behaviors or the recorded possible state transitions did not offer a learning structure for continuous development or adaptation to unexpected situations. This limitation hindered the ability of the robots to adapt to new or unforeseen conditions that might arise during locomotion, such as varying terrains or obstacles.

3. **Focus on Specific Objectives**: Many of the reinforcement learning applications in gait generation were primarily aimed at maximizing speed or achieving specific walking patterns. While these objectives are important, they often did not encompass the broader goal of generating free, adaptive, and efficient gaits that can handle a wide range of walking conditions and speeds.

4. **Complexity and Time Consumption in Real-time Applications**: The application of reinforcement learning to actual robotic systems in real-time was noted to be extremely difficult due to the necessity of a large number of training cycles. This complexity made it challenging to apply learning algorithms directly to robots without significant simplification or incorporation of a priori knowledge.

In contrast, the solution presented in the paper focuses on free gait generation with reinforcement learning that not only guarantees stability but also improves the adaptability and efficiency of gait generation in real-time. This approach addresses the adaptability to unexpected situations like leg deficiencies and enhances the continuous development structure for the robot's locomotion.
