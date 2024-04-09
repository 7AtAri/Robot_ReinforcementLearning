import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque # queue for efficiently adding and removing elements from both ends
import gymnasium as gym
from gymnasium.envs.registration import register
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import os

print ("GPU erkannt: " + str(torch.cuda.is_available())) # checks if gpu is found
# mute the MKL warning on macOS
#os.environ["MKL_DEBUG_CPU_TYPE"] = "5"

torch.set_default_dtype(torch.float)


class QNetworkCNN(nn.Module):
    def __init__(self, state_size, actions):
        super(QNetworkCNN, self).__init__()
        # Initialize convolutional and pooling layers
        self.conv1 = nn.Conv3d(in_channels=state_size[0], out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Use a dummy input to calculate flat features
        self.flat_features = self.calculate_flat_features(torch.zeros(1, state_size[0], state_size[1], state_size[2], state_size[3]).float())
        
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
    def __init__(self, state_size, actions, epsilon_decay, epsilon_min,device, lr=5e-4, gamma=0.99, batch_size=32, buffer_size=10000, n_step=3):
        self.state_size = state_size
        self.actions = actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)  # temporary buffer for n-step calculation
        self.n_step = n_step
        # this ensures that the memory does not grow beyond buffer_size - oldest elements are removed:
        self.memory = deque(maxlen=buffer_size) 
        
        self.q_network = QNetworkCNN(state_size, actions).to(device)
        self.target_network = QNetworkCNN(state_size, actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay#0.995  # 0.9 for debugging only
        self.epsilon_min = epsilon_min

    def add_experience(self, state, action, reward, next_state, done):
        # keep experience in n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))

        if len(self.n_step_buffer) == self.n_step or done:
            n_step_reward, n_step_state, n_step_done = self.calculate_n_step_info()
            # keep n-step transition in memory
            self.memory.append((self.n_step_buffer[0][0], self.n_step_buffer[0][1], n_step_reward, n_step_state, n_step_done))

            if done:
                # Clear the buffer if the episode ended
                self.n_step_buffer.clear()

    def calculate_n_step_info(self):
        """Calculate n-step reward, final state, and done status."""
        n_step_reward = 0
        for idx, (_, _, reward, _, _) in enumerate(self.n_step_buffer):
            n_step_reward += (self.gamma ** idx) * reward
        # The final state and done flag from the last experience
        _, _, _, n_step_state, n_step_done = self.n_step_buffer[-1]
        return n_step_reward, n_step_state, n_step_done


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
        action = action.tolist()
        print("action shape:", len(action))
        print("-------------------------------")
        return action


    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device).view(-1) # shape [batch_size]
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones)).to(device) # dones is terminated or truncated states
        # convert actions to long tensor for indexing
        actions = actions.long()
        # actions is of shape [batch_size], containing the index of the action taken for each batch item
        actions = actions.view(-1, 6, 1)  # reshape for gathering: [batch_size*6, 1]
        states = states.float().to(device)  # Ensure states is a FloatTensor 
        # get the q-values from the q-network for the current states
        Q_values = self.q_network(states) # this returns Q(s,a) for all actions a
        # get the q-values for the actions taken
        Q_expected = Q_values.gather(2, actions).squeeze(-1)  # --> shape [batch_size, 6]
            
        # get the q-values from the target network for the next states
        Q_values_next = self.target_network(next_states).detach()
        Q_targets_next = Q_values_next.max(dim=2)[0]  # max along the last dimension --> shape [batch_size, 6]
        
        # not terminated or truncated states:
        not_done = 1 - (dones)
        not_done = not_done.to(device).view(-1)  # Ensure shape [batch_size]

        # shape rewards and not_done to [batch_size, 6]
        rewards_expanded = rewards.unsqueeze(1).expand_as(Q_targets_next)  # Expands to [batch_size, 6]
        not_done_expanded = not_done.unsqueeze(1).expand_as(Q_targets_next)  # Expands to [batch_size, 6]

        ## n-step bootstrap state to calculate the next Q-values:
        Q_targets = rewards_expanded + (self.gamma**self.n_step * Q_targets_next * not_done_expanded)
            
        # compute loss, perform backpropagation and update weights:
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            print("epsilon reduced:", self.epsilon)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())



if __name__ == "__main__":
    grid = [{'batch_size': 16, 'episodes': 200, 'epsilon_decay': 0.9, 'epsilon_min': 0.25},
                {'batch_size': 8, 'episodes': 100, 'epsilon_decay': 0.95, 'epsilon_min': 0.1},
                {'batch_size': 8, 'episodes': 100, 'epsilon_decay': 0.995, 'epsilon_min': 0.2},
                {'batch_size': 16, 'episodes': 100, 'epsilon_decay': 0.9, 'epsilon_min': 0.2},
                {'batch_size': 32, 'episodes': 500, 'epsilon_decay': 0.95, 'epsilon_min': 0.2},
                {'batch_size': 64, 'episodes': 1000, 'epsilon_decay': 0.99, 'epsilon_min': 0.2},
                {'batch_size': 32, 'episodes': 200, 'epsilon_decay': 0.9, 'epsilon_min': 0.4},
                {'batch_size': 16, 'episodes': 300, 'epsilon_decay': 0.95, 'epsilon_min': 0.3},
                {'batch_size': 64, 'episodes': 1000, 'epsilon_decay': 0.995, 'epsilon_min': 0.1}]
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

    # # Training loop
    episodes = 0
    epsilon_decay = 0
    epsilon_min = 0

    for i in range(len(grid)):
        params = grid[i]
        for key, val in params.items():
            exec(key + '=val')   # assign the values to the hyperparameters
        # initialize the agent
        agent = DQNAgent(state_size, actions, epsilon_decay, epsilon_min, device)

        #agent = DQNAgent(state_size, actions, device=device)
        print(f"State size: {state_size}, Action size: {actions}")

        min_distances = [] # list to save the minum distanz of ech episode
        min_distance_tcp_helix = None
   
        for episode in range(episodes):
            state, info = env.reset()  
            #state = torch.FloatTensor(state).unsqueeze(0)  # add batch dimension
            terminated = False
            truncated = False
            step_counter = 0
            total_reward = 0

            prev_episode_steps = 0 # counter for max steps per episode
            while not terminated and not truncated:
                # state is the observation (1. voxel space with helix and 2. voxel space with TCP position) 
                action = agent.act(state)
                #print("action:", action)
                next_state, reward, terminated, truncated, info = env.step(action)  
                # if step_counter > 1:
                #     env.render()
                min_distance_tcp_helix = info['closest_distance']
                #print("next_state:", next_state)
                #print("next_state shape:", next_state.shape)
                #next_state = np.reshape(next_state, [1, state_size])
                agent.add_experience(state, action, reward, next_state, terminated or truncated)
                state = next_state
                total_reward += reward
                step_counter += 1
                print("total_reward", total_reward)
                print("terminated:", terminated)
                print("truncated:", truncated)

                if step_counter > prev_episode_steps:  # Check if the number of steps in the current episode is greater than the previous episode
                    if step_counter % 2 == 0:  # Check if the current step is a multiple of 5
                        env.render()
                        prev_episode_steps = step_counter  # Update the number of steps in the previous episode

            while len(agent.n_step_buffer) > 0:
                n_step_reward, n_step_state, n_step_done = agent.calculate_n_step_info()
                first_experience = agent.n_step_buffer.popleft()
                agent.memory.append((first_experience[0], first_experience[1], n_step_reward, n_step_state, n_step_done))
                
            min_distances.append(min_distance_tcp_helix) # same size as episode
            if terminated or truncated:
                print(f"Episode: {episode+1}/{episodes}, Total Reward: {total_reward}, Total Steps: {step_counter}, Epsilon: {agent.epsilon:.2f}")
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            agent.replay()
            if episode % 10 == 0:
                #env.render()
                agent.update_target_network()

        # calcualte mse for each episode --> first arg is expected distanz --> zero?
        mse = mean_squared_error(np.zeros(episodes), min_distances)
        # wenn for beednet wurde
        # loop draußen dann mse plot
        
        # Erstellen des Plots
        plt.plot(range(1, episodes + 1), mse, marker='o', linestyle='-')
        plt.xlabel('Episode')
        plt.ylabel('MSE')
        plt.title('Mean Squared Error (MSE) über Episoden')
        plt.grid(True)
        # Save the figure
        plt.savefig(os.path.join('Figure_1', 'MSE.png'))
        #plt.show()










