import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque # queue for efficiently adding and removing elements from both ends
import gymnasium as gym
from gymnasium.envs.registration import register
import torch.nn.functional as F

import os
# mute the MKL warning on macOS
os.environ["MKL_DEBUG_CPU_TYPE"] = "5"


class QNetworkCNN(nn.Module):
    def __init__(self, state_size, actions):
        super(QNetworkCNN, self).__init__()
        # Initialize convolutional and pooling layers
        self.conv1 = nn.Conv3d(in_channels=state_size[0], out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Use a dummy input to calculate flat features
        self.flat_features = self.calculate_flat_features(torch.zeros(1, state_size[0], state_size[1], state_size[2], state_size[3]))
        
        # Initialize fully connected layers using the calculated flat_features
        self.fc1 = nn.Linear(self.flat_features, 256)
        self.fc2 = nn.Linear(256, actions*3)   # 3 possible values per action component 

    def forward(self, x):
        # Define the forward pass
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1, 6, 3)
 
    def calculate_flat_features(self, dummy_input):
        # Pass the dummy input through the convolutional and pooling layers
        with torch.no_grad():
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.pool(F.relu(self.conv2(x))) 
        # The output's total size is the required number of flat features
        return x.numel() // x.shape[0]  # Divide by batch size (x.shape[0]) to get the size per sample
    


class DQNAgent:
    def __init__(self, state_size, actions, lr=5e-4, gamma=0.99, batch_size=8, buffer_size=10000):
        self.state_size = state_size
        self.actions= actions
        self.batch_size = batch_size
        self.gamma = gamma
        # this ensures that the memory does not grow beyond buffer_size - oldest elements are removed:
        self.memory = deque(maxlen=buffer_size) 
        
        self.q_network = QNetworkCNN(state_size, actions).to(device)
        self.target_network = QNetworkCNN(state_size, actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        self.epsilon = 1.0
        self.epsilon_decay = 0.9 #0.995  # 0.9 for debugging only
        self.epsilon_min = 0.01

    def remember(self, state, action, reward, next_state, terminated, truncated):
        self.memory.append((state, action, reward, next_state, terminated, truncated))
        
    # def act(self, state):
    #     if np.random.rand() <= self.epsilon:
    #         return random.randrange(self.action_size)
    #     state = torch.FloatTensor(state).unsqueeze(0)
    #     q_values = self.q_network(state)
    #     return np.argmax(q_values.detach().numpy()) 

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # return random action for each component
            action = [random.randrange(3) for _ in range(6)]
            print("exploring: random action")
            print("action shape:", len(action))
            return action # shape [6] ?
            
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = self.q_network(state)
        print("exploiting: q-values predicted from network") # q-values: torch.Size([1, 6, 3])
        
        # choose action with max Q-value for each component
        action = q_values.detach().cpu().numpy().argmax(axis=2).flatten() 
        print("action shape:", action.shape)
        print("-------------------------------")
        return action


    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, terminated, truncated = zip(*minibatch)
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        rewards = torch.FloatTensor(rewards).to(device).view(-1) # shape [batch_size]
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        terminated = torch.FloatTensor(np.array(terminated)).to(device)
        truncated = torch.FloatTensor(np.array(truncated)).to(device)
        print("actions:", actions.shape) # actions: torch.Size([batch_size, 6])
        # actions tensor must have the correct shape and type
        actions = actions.long()  # long type for indexing
        # Assuming actions is of shape [batch_size], containing the index of the action taken for each batch item
        actions = actions.view(-1, 6, 1)  # Reshape for gathering: [batch_size*6, 1]
        print("actions shape:", actions.shape)
        # using gather to select the action values:
        #  https://pytorch.org/docs/stable/generated/torch.gather.html
        # we need to gather along the last dimension (dimension=2) of the Q-values tensor
        #Q_expected = self.q_network(states).gather(2, actions.unsqueeze(-1)).squeeze(-1)
        #Q_expected = self.q_network(states).gather(1, actions)  # This selects the Q-values for the taken actions, resulting in shape [64, 1]
        #Q_expected = Q_expected.squeeze()  # Remove the last dimension to match Q_targets: shape becomes [64]
        # Forward pass on the current states to get Q values for all actions
        Q_values = self.q_network(states)
        print("q-values:", Q_values.shape) # q-values: torch.Size([batch_size, 6, 3])
        # Select the Q-values for the actions taken
        Q_expected = Q_values.gather(2, actions).squeeze(-1)  # --> shape [batch_size, 6]
        print("q-expected:", Q_expected.shape)
        #Q_expected = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        #Q_targets_next = self.target_network(next_states).detach().max(1)[0]
        Q_values_next = self.target_network(next_states).detach()
        print("q-values next:", Q_values_next.shape) # q-values next: torch.Size([batch_size, 6, 3])
        Q_targets_next = Q_values_next.max(dim=2)[0]  # max along the last dimension --> shape [batch_size, 6]
        #Q_values_flattened = Q_values_next.view(self.batch_size, -1)
        #Q_targets_next = Q_values_flattened.max(dim=1)[0]# should output a [batch_size] tensor
        print("q-targets next:" , Q_targets_next.shape)
        # if episode was either terminated or truncated, we don't look at the next state's Q-value
        not_done = 1 - (terminated + truncated)
        not_done = not_done.to(device).view(-1)  # Ensure shape [batch_size]
        print("reward shape:", rewards.shape)
        print("not done shape:", not_done.shape)
        # Expand rewards and not_done to enable broadcasting
        rewards_expanded = rewards.unsqueeze(1).expand_as(Q_targets_next)  # Expands to [batch_size, 6]
        not_done_expanded = not_done.unsqueeze(1).expand_as(Q_targets_next)  # Expands to [batch_size, 6]

        # Now compute Q_targets with correctly shaped tensors
        Q_targets = rewards_expanded + (self.gamma * Q_targets_next * not_done_expanded)

        # Q_expected: expected future rewards (of the network) for the chosen action 
        # Q_targets: actual future rewards (from the target network) for the chosen action
        loss = nn.MSELoss()(Q_expected, Q_targets) 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            print("epsilon reduced:", self.epsilon)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())



if __name__ == "__main__":
    # check which device is available
    #device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    register(
        id='RobotEnvironment-v1',
        entry_point='Environment-cnn:RobotEnvironment',
    )

    env = gym.make('RobotEnvironment-v1')
    print("obs space:", env.observation_space.shape)
    state_size = env.observation_space.shape  # (2, 61, 61, 101)
    actions = env.action_space.shape[0] #.nvec.prod()  # actions = 6
    agent = DQNAgent(state_size, actions)
    print(f"State size: {state_size}, Action size: {actions}")

    # # Training loop
    episodes = 20
    for episode in range(episodes):
        state, info = env.reset()  
        #state = torch.FloatTensor(state).unsqueeze(0)  # add batch dimension
        terminated = False
        truncated = False
        step_counter = 0
        total_reward = 0
        while not terminated and not truncated:
            # state is the observation (1. voxel space with helix and 2. voxel space with TCP position) 
            action = agent.act(state)
            #print("action:", action)
            next_state, reward, terminated, truncated, _ = env.step(action)  
            step_counter += 1
            #print("next_state:", next_state)
            #print("next_state shape:", next_state.shape)
            #next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, terminated, truncated)
            state = next_state
            total_reward += reward
            print("total_reward", total_reward)
            print("terminated:", terminated)
            print("truncated:", truncated)
        if terminated or truncated:
            print(f"Episode: {episode+1}/{episodes}, Total Reward: {total_reward}, Total Steps: {step_counter}, Epsilon: {agent.epsilon:.2f}")
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        agent.replay()
        if episode % 10 == 0:
            env.render()
            agent.update_target_network()