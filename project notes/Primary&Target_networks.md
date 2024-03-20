# Using Primary and Target network in Deep Q-Learning

In the original formulation of Q-Learning and its initial deep learning adaptations, a single network was used for both selecting actions and evaluating the policy. However, this approach can lead to unstable training dynamics and poor policy convergence because it suffers from a key issue: the moving target problem. In Q-Learning, the Q-value update rule uses the same Q-values to estimate both the current (policy) and maximum future rewards. When these Q-values are constantly changing due to ongoing learning, it creates a feedback loop that can lead to oscillations or divergence in the value estimates.

The innovation of using a separate primary (online) network and a target network was introduced to mitigate this problem, and it has become a standard practice in many Deep Q-Learning algorithms, such as the Deep Q-Network (DQN). Here's how the two networks function within this framework:

    - Primary (Online) Network: This network is updated at every step or after a set number of steps during training. It directly interacts with the environment by evaluating the current policy and choosing actions. The weights of this network are adjusted continuously through backpropagation based on the loss calculated from the difference between predicted Q-values and the target Q-values.

    - Target Network: The target network is a copy of the primary network that is kept static for a certain number of steps. Its main role is to generate stable target values for the Q-value updates. Periodically, the weights of the primary network are copied to the target network to synchronize them. This separation helps in stabilizing the learning process by providing fixed targets for a while.

The use of both networks addresses the moving target problem by decoupling the target value generation from the weights being updated. This approach significantly improves the stability and convergence of the learning process in complex environments.

However, it's essential to note that while the dual-network architecture is a common and effective strategy in Deep Q-Learning, not all variations of Q-Learning or deep reinforcement learning algorithms use this exact approach. For instance, some algorithms may employ techniques like experience replay without necessarily using a separate target network, or they may use other methods to stabilize the training process. The specific choice of architecture and techniques depends on the algorithm's design and the problem being addressed.

## Learning a policy in a deep q-learning network with primary and target part and n-step bootstrapping:

The policy improvement in the provided DQNAgent code occurs implicitly through the updates made to the Q-network (the "model" in the code). Let's break down where and how this happens:

### Q-Value Prediction and Action Selection

The core of the DQN algorithm is the Q-network, which predicts the Q-values for each action given the current state of the environment. The Q-value represents the expected cumulative reward that the agent can obtain, starting from the current state and taking a particular action followed by an optimal policy. Therefore, the action with the highest Q-value in a given state is considered the best action according to the current policy.

### Epsilon-Greedy Strategy for Exploration and Exploitation

The `act` method implements an epsilon-greedy strategy, where with probability `epsilon`, the agent selects a random action (exploration), and with probability `1 - epsilon`, it selects the action with the highest Q-value predicted by the Q-network (exploitation). This approach allows the agent to explore the environment and discover new strategies while exploiting its current knowledge to maximize the reward.

### Policy Improvement through Learning

The policy improvement occurs during the learning phase in the `learn` method. Here's the step-by-step process:

1. **Sampling from Replay Memory**: The agent samples a batch of experiences (state, action, reward, next_state, done) from the replay memory. These experiences are used to update the Q-network.

2. **Calculating Target Q-values**: For each experience in the batch, the agent calculates the target Q-value using the reward from the experience and the predicted Q-values from the target network for the next state. The target Q-value for a state-action pair \((s, a)\) is computed as:
   $
   \text{target} = r + \gamma \max_{a'} Q_{\text{target}}(s', a')
   $
   where $r$ is the reward, $\gamma$ is the discount factor, $s'$ is the next state, and $\max_{a'} Q_{\text{target}}(s', a')$ is the maximum Q-value predicted by the target network for the next state. This represents the Bellman equation for the Q-value, aiming to approximate the optimal policy.

3. **Updating the Q-network**: The agent then updates the Q-network (the model) to minimize the difference (loss) between its current Q-value predictions and these target Q-values. The loss function used is often Mean Squared Error (MSE) or Huber loss. By minimizing this loss, the Q-network learns to predict Q-values that are closer to the true Q-values of the optimal policy.

4. **Synchronizing the Target Network**: Periodically, the weights of the Q-network are copied to the target network. This step ensures that the target Q-values used for training gradually adapt as the Q-network learns and improves.

The policy is implicitly improved through this learning process. As the Q-network gets better at predicting Q-values that reflect the true value of taking actions in different states, the action selection (exploitation part of the epsilon-greedy strategy) naturally becomes more effective, leading to a better policy. This iterative process of policy evaluation (predicting Q-values) and policy improvement (updating the network based on better target Q-values) underlies many reinforcement learning algorithms, including DQN.
