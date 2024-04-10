# N-Step Bootstrapping vs Primary and Target Networks

The concepts of using a primary and target network in Deep Q-Learning and n-step bootstrapping are related in their goal to stabilize and improve the training process of reinforcement learning algorithms, but they address different aspects of the learning process. Here’s how they differ and relate:

Primary and Target Networks

- Purpose: The primary network predicts Q-values for state-action pairs, guiding the agent's decision-making. The target network generates stable target Q-values for the loss calculation, reducing the correlation between the Q-value being updated and the target Q-value. This setup helps mitigate the moving target problem, where updates might chase a constantly shifting goal due to the rapid changes in the estimate itself.
- Mechanism: Periodically, the weights of the primary network are copied to the target network to keep the targets somewhat stable yet allow them to gradually adapt as the primary network learns.

N-Step Bootstrapping

- Purpose: N-step bootstrapping is a technique to adjust the balance between bias and variance in the estimate of future rewards. Instead of relying solely on the next immediate reward plus the value of the subsequent state (as in 1-step returns) or the cumulative reward over the entire episode (as in Monte Carlo approaches), n-step bootstrapping takes an intermediate approach. It considers the sum of rewards over n steps plus the estimated value of the state n steps ahead.
- Mechanism: This method aims to speed up learning by providing a more informative signal about the outcome of actions over multiple steps, potentially leading to faster policy improvement. It strikes a balance between the often high variance of Monte Carlo methods (which wait until the end of an episode to update value estimates) and the high bias of methods that heavily rely on current value estimates (like 1-step Q-Learning).

Relationship and Differences

- Stabilization vs. Speed and Balance: The primary and target network strategy primarily addresses the stability of learning by mitigating the moving target problem. In contrast, n-step bootstrapping seeks to improve learning speed and the balance between bias and variance in value estimation.
- Complementary Use: These techniques can be complementary. For example, a Deep Q-Learning algorithm could use both an n-step return approach to compute the target Q-values and a target network to provide stability in those targets. This combination can enhance the efficiency and effectiveness of the learning process.

In summary, while both techniques aim to improve the reinforcement learning process, they do so from different angles: one focuses on stabilizing the training process, and the other on improving the estimates of future rewards. Their combined use can potentially offer a more robust learning algorithm.

## Mathematical Formulation of n-step-bootstrapping for Deep Q-Learning

- computing the n-step return from a sequence of experiences stored in an n-step buffer
- n-step return as target for updating the Q-value estimates in Deep Q-Learning

each experience at step i consists of the current state $s_{i}$,  action $a_{i}$, reward $r_{i}$, and the discount factor $\gamma$, the n-step return, $R_{t:t+n}$, can be calculated like this:

$$
R_{t:t+n} = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \ldots + \gamma^{n-1} r_{t+n-1}
$$

Episode continues for at least n steps from time t.

To update the Q-value:
$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[R_{t:t+n} + \gamma^n \max_{a}Q(s_{t+n}, a) - Q(s_t, a_t)\right]
$$

$\alpha$ is the learning rate, $\max_{a}Q(s_{t+n}, a)$ represents the value of the best possible action in the state $s_{t+n}$.This term is omitted if the episode ends within the n steps, as indicated by the done flag.

This formula encapsulates the essence of n-step bootstrapping in Q-learning, aiming to balance immediate feedback with longer-term outcomes for more effective learning.
