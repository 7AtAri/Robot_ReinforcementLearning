import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque # queue for efficiently adding and removing elements from both ends
import gymnasium as gym
from gymnasium.envs.registration import register

import os
# mute the MKL warning on macOS
os.environ["MKL_DEBUG_CPU_TYPE"] = "5"

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(QNetwork, self).__init__()
        # Adjust hidden layer size as necessary
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.relu = nn.ReLU()
        # Adjust output layer to match the total number of actions (6 actions * 3 possibilities each)
        self.fc2 = nn.Linear(hidden_size, action_size * 3)
        
    def forward(self, state):
        x = self.relu(self.fc1(state))
        return self.fc2(x).view(-1, 6, 3)  # Reshape output to have 3 values per action component



from collections import deque
import random

class DQNAgent:
    def __init__(self, state_size, action_size, lr=5e-4, gamma=0.99, batch_size=64, buffer_size=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # def act(self, state):
    #     if np.random.rand() <= self.epsilon:
    #         return random.randrange(self.action_size)
    #     state = torch.FloatTensor(state).unsqueeze(0)
    #     q_values = self.q_network(state)
    #     return np.argmax(q_values.detach().numpy()) 

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # Return a random action for each component
            return [random.randrange(3) for _ in range(6)]
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state)
        # Choose the action with max Q-value for each component
        return q_values.detach().numpy().argmax(axis=2).flatten()


    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        Q_expected = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        Q_targets_next = self.target_network(next_states).detach().max(1)[0]
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        loss = nn.MSELoss()(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())



if __name__ == "__main__":
    # check which device is available
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    register(
        id='RobotEnvironment-v0',
        entry_point='Environment:RobotEnvironment',
    )

    env = gym.make('RobotEnvironment-v0')
    state_size = np.prod(env.observation_space.shape)  # State size: 375821
    action_size = env.action_space.nvec.prod() # Action size: 729
    agent = DQNAgent(state_size, action_size)

    print(f"State size: {state_size}, Action size: {action_size}")

    # # Training loop
    # episodes = 10
    # for e in range(episodes):
    #     state, info = env.reset()
    #     # extract the initial TCP position from the reset info dictionary
    #     tcp_position = info['tcp_position']
    #     state = np.reshape(state, [1, state_size])
    #     done = False
    #     total_reward = 0
    #     while not done:
    #         action = agent.act(state)
    #         next_state, reward, terminated, truncated, info = env.step(action)  # Adjust according to your env's step method
    #         next_state = np.reshape(next_state, [1, state_size])
    #         agent.remember(state, action, reward, next_state, done)
    #         state = next_state
    #         total_reward += reward
    #     if done:
    #                 print(f"Episode: {e+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
    #     agent.replay()
    #     if e % 10 == 0:
    #         agent.update_target_network()
