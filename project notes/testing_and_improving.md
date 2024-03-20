# Testing and Validation

## Problem

The success of DQN training can be highly sensitive to the choice of hyperparameters, including learning rate, discount factor (γ), memory size, batch size, and the frequency of target network updates.

## solution

- Hyperparameter Tuning

- Experiments with different settings to find a combination that works well

- validation runs without exploration (epsilon = 0) to evaluate the performance of the learned policy without noisy explorations
