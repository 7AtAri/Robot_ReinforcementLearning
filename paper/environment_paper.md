The reinforcement learning (RL) environment used in the paper described using mathematical terms:

1. **State Space (S)**: The state of the robot is defined by the discrete positions of all six legs, represented as a vector $\(P = [s_1, s_2, s_3, s_4, s_5, s_6]\)$, where each $\(s_i\)$ can take values indicating the position of the leg (either in a supporting position labeled 1 to 5 or in a return position labeled 0). This means in total there are $6^{6}$ theoretical states. But among those there are also dead-end states and impossible states, which are excluded by the algorithm.

2. **Action Space (A)**: Actions in this environment are the possible transitions between states, dictated by the movements of the legs. The action a robot can take from a state $P$ to a next state $P'$ depends on the current configuration of the legs and the commanded velocity $v_{unit}$. The FSG (Free State Generation) algorithm and modifications for rear-leg deficiency dynamically generate the available actions based on the current state and conditions.

3. **Reward Function (R)**: The reward $r$ is assigned after each transition based on the stability margin of the resulting state and the commanded velocity. The reward function aims to encourage transitions to states with larger stability margins and penalize transitions that lead to falls or unstable configurations. The reward $r_{sm}(n)$ for moving from state $P(n)$ to $P(n+1)$ at iteration $n$ is calculated as:

$$
r_{\text{sm}}(n) = 
\begin{cases} 
\frac{1}{1 + e^{(7-sm_{n+1})}} - 0.5 & \text{if } sm_{n+1} < 7 \\
1 - \frac{1}{1 + e^{(sm_{n+1}-7)}} & \text{otherwise}
\end{cases}
$$


Where $sm_{n+1}$ is the stability margin of the next state $P(n+1)$.
This function rewards transitions that lead to a state with a stability margin greater than or equal to 7 cm, and penalizes transitions resulting in lower stability margins.

4. **Policy (π)**: The policy is a strategy that the learning algorithm uses to decide the actions to take based on the current state. It aims to maximize the cumulative reward. The FGGRL (Free Gait Generation with Reinforcement Learning) dynamically updates the policy based on the observed rewards, utilizing reinforcement learning to favor transitions that have historically led to stable and efficient gaits.

5. **Learning Mechanism**: The learning mechanism updates the utilities (values) of state transitions based on the reinforcement signals received after each action. The utilities are used to inform the selection of actions (state transitions) in future iterations, favoring those that lead to higher rewards. The update rule for the utility $u_{P(n) \rightarrow P(n+1)}$ of transitioning from state $P(n)$ to $P(n+1)$ is influenced by the received reward $r_{sm}(n)$ and a discount factor $\gamma$, emphasizing the importance of recent transitions over older ones. P(n) is initialized to 0.5.

$$ u_{P(n) \to P(n+1)} := u_{P(n) \to P(n+1)} + \gamma_{i-n}*r(n)* [1 - u_{P(n) \to P(n+1)}] $$

for $i=n-4, n-3,...,n$

6. **Evaporation Mechanism (Discounting Over Time)**:
   
$$ u_{P(i) \to P(i+1)} := \beta \cdot u_{P(i) \to P(i+1)} $$

This formula represents the evaporation mechanism where $\beta$  = 0.9 acts similarly to a discount factor but is applied across all utilities globally at regular intervals (e.g., after each iteration or epoch), rather than being tied to the calculation of future rewards as in traditional reinforcement learning models.
The purpose of this mechanism is to gradually decrease the value of all stored utilities over time, effectively reducing the importance of older transitions and making the algorithm more responsive to new experiences. $\beta$ is set to 0.9, indicating that after each iteration, 90% of each utility value is retained, and 10% is "evaporated."

The key distinction here is that $\gamma$ directly influences how much new reward observations affect the utility of a specific transition, focusing on the integration of new knowledge. In contrast, $\beta$ serves as a mechanism for forgetting or de-emphasizing older information across all transitions, ensuring that the learning process does not become overly reliant on historical data that may no longer be relevant. Both parameters are crucial for balancing exploration and exploitation, as well as adapting to changing conditions or objectives within the learning environment.

### Decision Process for Actions
The next state is chosen based on the memorized transitions with the highest utility or by generating a new state through the free gait generation algorithm. The probability of using a memorized transition is proportional to the utilities of the available transitions from the current state.

### Handling Rear-Leg Deficiency
For the case of rear-leg deficiency, the learning process focuses on finding quasi-statically stable transitions, adjusting the reward function to reflect the ability to maintain stability despite the deficiency.
This mathematical framework forms the basis of the reinforcement learning environment used for free gait generation and adaptation in a six-legged robot, allowing the robot to learn and improve its walking patterns dynamically.
