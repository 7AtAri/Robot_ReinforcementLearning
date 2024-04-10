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

import datetime
import os
import shutil

print ("GPU erkannt: " + str(torch.cuda.is_available())) # checks if gpu is found
# mute the MKL warning on macOS
#os.environ["MKL_DEBUG_CPU_TYPE"] = "5"

torch.set_default_dtype(torch.float)

class LogStore():
    """_summary_
    This class is used to store the log information in a text file.

    _attributes_
    filename: str
        The name of the file to store the log information
    
    _methods_
    setfilename(name: str)
        Set the filename to store the log information
    write_to_log(input: str)
        Write the input to the log file
    """
    def __init__(self):
        self.filename = ""

    def setfilename(self,name):
        self.filename = name

    def write_to_log(self, input):
            with open("code/" + self.filename + ".txt", 'a') as log:
                log.write(input + "\n")


class QNetworkCNN(nn.Module):
    """_summary_
    This class defines the Q-network 
    using a Multimodal Approach: Convolutional Neural Network (CNN) architecture for the spatial data
    and Late Feature Fusion for the TCP orientation data to predict the Q-values for the actions.

    _attributes_
    conv1 & conv2: nn.Conv3d
        3-Dimensionaler Convolutional layer 
    pool: nn.MaxPool3d
        3-Dimensionaler MaxPooling layer
    flat_features: int
        The number of flat features calculated from the convolutional layers
    fc1 & fc2: nn.Linear
        Fully connected layers
    
    _methods_
    forward(x, tcp_data)
        Forward pass through the network
    calculate_flat_features(dummy_input)
        Calculate the number of flat features from the convolutional layers
    """
    def __init__(self, state_size, actions, tcp_feature_size=3):
        super(QNetworkCNN, self).__init__()
        # Initialize convolutional and pooling layers
        self.conv1 = nn.Conv3d(in_channels=state_size[0], out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Use a dummy input to calculate flat features
        self.flat_features = self.calculate_flat_features(torch.zeros(1, state_size[0], state_size[1], state_size[2], state_size[3]).float())
        
        # Initialize fully connected layers using the calculated flat_features
        self.fc1 = nn.Linear(self.flat_features + tcp_feature_size, 256)
        self.fc2 = nn.Linear(256, actions*3)   # 3 possible values per action component 

    def forward(self, x, tcp_data):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        # concatenate flattened spatial data with TCP position and orientation data
        #print("x shape:", x.shape)
        #print("tcp_data shape:", tcp_data.shape)
        tcp_data= torch.squeeze(tcp_data, dim=1)
        combined_features = torch.cat((x, tcp_data), dim=1)

        x = F.relu(self.fc1(combined_features))
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
    """_summary_
    This class defines the DQN agent that interacts with the environment.
    The agent uses a Q-network to predict the Q-values for the actions and learns from the experiences in the memory.

    _attributes_
    spatial_data_shape: tuple
        The shape of the spatial data
    actions: int
        The number of actions
    epsilon: float
        The exploration rate
    epsilon_decay: float
        The rate at which the epsilon value decays
    epsilon_min: float
        The minimum value of epsilon
    device: str
        The device to run the computations on
    lr: float
        The learning rate
    gamma: float
        The discount factor
    batch_size: int
        The batch size
    buffer_size: int
        The size of the memory buffer
    n_step: int
        The number of steps to calculate the n-step return
    n_step_buffer: deque
        A temporary buffer for n-step calculation
    memory: deque
        The memory buffer to store experiences
    q_network: QNetworkCNN
        The Q-network to predict Q-values
    target_network: QNetworkCNN
        The target network to predict Q-values
    optimizer: optim.Adam
        The optimizer to update the weights of the Q-network

    _methods_
    add_experience(spatial_data, tcp_data, action, reward, next_spatial_data, next_tcp_data, done)
        Add the experience to the memory
    calculate_n_step_info()
        Calculate the n-step reward, final state, and done status
    act(state)
        Choose an action based on the epsilon-greedy policy
    replay()
        Learn from the experiences in the memory
    update_target_network()
        Update the target network with the weights of the Q-network
    """

    def __init__(self, spatial_data_shape, actions, epsilon_decay, epsilon_min,device, lr=5e-4, gamma=0.99, batch_size=32, buffer_size=10000, n_step=3):
        self.spatial_data_shape = spatial_data_shape
        self.actions = actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)  # temporary buffer for n-step calculation
        self.n_step = n_step
        # this ensures that the memory does not grow beyond buffer_size - oldest elements are removed:
        self.memory = deque(maxlen=buffer_size) 
        
        self.q_network = QNetworkCNN(self.spatial_data_shape, actions, tcp_feature_size=3).to(device)
        self.target_network = QNetworkCNN(self.spatial_data_shape, actions, tcp_feature_size=3).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay#0.995  # 0.9 for debugging only
        self.epsilon_min = epsilon_min

    def add_experience(self, spatial_data, tcp_data, action, reward, next_spatial_data, next_tcp_data,done):
        # keep experience in n-step buffer
        experience = ((spatial_data, tcp_data), action, reward, (next_spatial_data, next_tcp_data), done)
        self.n_step_buffer.append(experience)

        if len(self.n_step_buffer) == self.n_step or done:
            n_step_reward, n_step_state, n_step_done = self.calculate_n_step_info()
            # keep n-step transition in memory
            self.memory.append((self.n_step_buffer[0][0], self.n_step_buffer[0][1], n_step_reward, n_step_state, n_step_done))

            if done:
                # Clear the buffer if the episode ended
                self.n_step_buffer.clear()

    def calculate_n_step_info(self):
        """_summary_
        Calculate the n-step reward, final state, and done status

        _returns_
        n_step_reward: float
            The n-step reward
        n_step_state: tuple
            The final state
        n_step_done: bool
            The done status
        """
        n_step_reward = 0
        for idx, (_, _, reward, _, _) in enumerate(self.n_step_buffer):
            n_step_reward += (self.gamma ** idx) * reward
        # The final state and done flag from the last experience
        _, _, _, n_step_state, n_step_done = self.n_step_buffer[-1]
        return n_step_reward, n_step_state, n_step_done


    def act(self, state):
        """_summary_
        Choose an action based on the epsilon-greedy policy

        _parameters_
        state: tuple
            The current state

        _returns_
        action: list
            The chosen action
        exploiting: bool
            The exploitation status
        """

        exploiting  =  False
        spatial_data, tcp_data = state  # unpack the state tuple
        if np.random.rand() <= self.epsilon:
            # return random action for each component
            action = [random.randrange(3) for _ in range(6)]
            print("exploring: random action")
            #print("action shape:", len(action))
            return action, exploiting # shape [6] ?

        spatial_data = torch.FloatTensor(spatial_data).unsqueeze(0).to(device)
        tcp_data = torch.FloatTensor(tcp_data).unsqueeze(0).to(device)
        q_values = self.q_network(spatial_data, tcp_data)
        print("exploiting: q-values predicted from network") # q-values: torch.Size([1, 6, 3])
        exploiting = True
        # choose action with max Q-value for each component
        action = q_values.detach().cpu().numpy().argmax(axis=2).flatten() 
        action = action.tolist()
        #print("action shape:", len(action))
        print("-------------------------------")
        print("action:", action)
        print("exploiting:", exploiting)
        return action, exploiting


    def replay(self):
        """_summary_
        Learn from the experiences in the memory
        """
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        #states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device).view(-1) # shape [batch_size]
        #next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones)).to(device) # dones is terminated or truncated states
        # convert actions to long tensor for indexing
        actions = actions.long()
        # actions is of shape [batch_size], containing the index of the action taken for each batch item
        actions = actions.view(-1, 6, 1)  # reshape for gathering: [batch_size*6, 1]
        #states = states.float() # Ensure states is a FloatTensor 

        # states includes both spatial and TCP data
        # spatial_data, tcp_data = states  # This line might need to be adjusted based on your data storage format
        # spatial_data = torch.stack(spatial_data).to(device)  # Convert list of tensors to a single tensor
        # tcp_data = torch.stack(tcp_data).to(device)  # Convert list of tensors to a single tensor

        # separate spatial and TCP data for states and next_states
        spatial_data = torch.stack([torch.FloatTensor(state[0]) for state in states]).to(device)
        tcp_data = torch.stack([torch.FloatTensor(state[1]) for state in states]).to(device)
        
        next_spatial_data = torch.stack([torch.FloatTensor(state[0]) for state  in next_states]).to(device)
        next_tcp_data = torch.stack([torch.FloatTensor(state[1]) for state in next_states]).to(device)

        # get the q-values from the q-network for the current states
        Q_values = self.q_network(spatial_data, tcp_data) # this returns Q(s,a) for all actions a
        # get the q-values for the actions taken
        Q_expected = Q_values.gather(2, actions).squeeze(-1)  # --> shape [batch_size, 6]
            
        # get the q-values from the target network for the next states
        Q_values_next = self.target_network(next_spatial_data, next_tcp_data).detach()
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
            #print("epsilon reduced:", self.epsilon)

    def update_target_network(self):
        """_summary_
        Update the target network with the weights of the Q-network
        """
        self.target_network.load_state_dict(self.q_network.state_dict())



if __name__ == "__main__":

    # grid for hyperparameters grid search
    grid = [{'batch_size': 8, 'episodes': 20, 'epsilon_decay': 0.9, 'epsilon_min': 0.25},
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

    # delete the plot folders if they already exist before starting the training
    folder_path = 'ParamCombi1'
    # Check if the folder exists and is a directory
    if os.path.isdir(folder_path):
        shutil.rmtree('ParamCombi1')

    folder_path = 'ParamCombi2'
    # Check if the folder exists and is a directory
    if os.path.isdir(folder_path):
        shutil.rmtree('ParamCombi2')

    # instantiate logging
    log = LogStore()
    log.setfilename("Setup")

    # register the environment
    register(
        id='RobotEnvironment-v1',
        entry_point='Environment-cnn:RobotEnvironment',
    )

    env = gym.make('RobotEnvironment-v1')

    # the shapes of the components of the Tuple space
    spatial_data_shape = env.observation_space[0].shape   # (2, 61, 61, 101)
    tcp_data_shape = env.observation_space[1].shape  # (6,)

    #log.write_to_log("obs space: " + str(env.observation_space[0].shape) + str(env.observation_space[1].shape))
    #print("Spatial Data Shape:", spatial_data_shape)
    #print("TCP Data Shape:", tcp_data_shape)

    actions = env.action_space.shape[0] # actions = 6
    
    # hyperparameters initialization
    episodes = 0
    epsilon_decay = 0
    epsilon_min = 0

     # # Training loop
    for i in range(len(grid)):
        params = grid[i]
        log.setfilename("Grid_" + str(i) +"_"+ str(datetime.datetime.today().strftime("%A_%H_%M")))
        log.write_to_log("-----------------------------------------------------------------------------------------------------------------------------------------------")
        log.write_to_log("Tested Parameters: " + str(params))
        for key, val in params.items():
            exec(key + '=val')   # assign the values to the hyperparameters

        # initialize the agent
        agent = DQNAgent(spatial_data_shape, actions, epsilon_decay, epsilon_min, device)

        #print(f"spatial_data_shape: {spatial_data_shape}, Action size: {actions}")

        # initialize the mse list
        mse_list = []
   
        for episode in range(episodes):
            state, info = env.reset()  # reset the environment and get the initial state
            terminated = False
            truncated = False
            step_counter = 0
            total_reward = 0
            log.write_to_log("+++++++++++++++++++++++++++Start Episode+++++++++++++++++++++++++++++++")
            while not terminated and not truncated:
                # state is the observation (1. voxel space with helix and 2. voxel space with TCP position) 
                action, exploiting = agent.act(state)

                # get results for the action from the environment:
                next_state, reward, terminated, truncated, info = env.step(action)  

                # get the information from the environment
                min_distance_tcp_helix = info['closest_distance']
                closest_helix_point = info['closest_point']
                current_tcp_position_voxels = info['tcp_position_in_voxels']
                current_tcp_orientation = info['current_orientation']
                tcp_on_helix = info['tcp_on_helix']
                current_tcp_position_coordinates = info['tcp_position_in_coordinates']

                # add the experience to the agent's memory
                agent.add_experience(*state, action, reward, *next_state, terminated or truncated)
                state = next_state # update to the current state
                total_reward += reward
                step_counter += 1

                #print("total_reward", total_reward)
                #print("terminated:", terminated)
                #print("truncated:", truncated)

                # log the information
                log.write_to_log(f"Exploiting: {exploiting}")
                log.write_to_log(f"current TCP Position in voxels: {np.round(current_tcp_position_voxels,6)}")
                log.write_to_log(f"Min Distance to Helix: {np.round(min_distance_tcp_helix,6)}")
                log.write_to_log(f"TCP on Helix: {tcp_on_helix}")
                log.write_to_log(f"Current TCP Orientation: {np.round(current_tcp_orientation, 2)}")
                log.write_to_log(f"Total Reward: {total_reward}")
                #log.write_to_log(f"Closest Helix Point: {np.round(closest_helix_point, 6)}")
                #log.write_to_log(f"current TCP Position in coordinates: {np.round(current_tcp_position_coordinates,6)}")

                # render the environment:
                if step_counter % 7 == 0:  # every 7 steps
                    env.render()
                    
            
            while len(agent.n_step_buffer) > 0:
                #print("----writing n-step buffer to memory-----")
                n_step_reward, n_step_state, n_step_done = agent.calculate_n_step_info()
                first_experience = agent.n_step_buffer.popleft()
                agent.memory.append((first_experience[0], first_experience[1], n_step_reward, n_step_state, n_step_done))

            if terminated or truncated:
                log.write_to_log(f"Episode: {episode+1}/{episodes}, Total Reward: {total_reward}, Total Steps: {step_counter}, Epsilon: {agent.epsilon:.2f}")
                #log.write_to_log("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print(f"Episode: {episode+1}/{episodes}, Total Reward: {total_reward}, Total Steps: {step_counter}, Epsilon: {agent.epsilon:.2f}")
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

            # replay the agent
            # this means that the agent learns from the experiences in the memory
            agent.replay()

            if episode % 10 == 0:
                #env.render()
                agent.update_target_network()

            # calcualte mse for each episode --> first arg is expected distanz --> zero?
            mse = mean_squared_error(current_tcp_position_coordinates, closest_helix_point)
            mse_list.append(mse)

        # log the end
        log.write_to_log("-----------------------------------------------------------------------------------------------------------------------------------------------")
        log.write_to_log("-------------------------------------End of Training with Parameter Combination-----------------------------------------------")
        
        # mse plot
        plt.figure()
        plt.plot(range(1, episodes + 1), mse_list, marker='o', linestyle='-')
        plt.xlabel('Episode')
        plt.ylabel('MSE')
        plt.title('Mean Squared Error (MSE) über Episoden')
        plt.grid(True)

        # check in which folder the file should be saved
        # check if one of the folders contains the MSE file:
        folder_path1 = 'ParamCombi1'
        folder_path2 = 'ParamCombi2'
        filename = 'MSE.png'
        # Construct the full path to the file
        file_path1 = os.path.join(folder_path1, filename)
        file_path2 = os.path.join(folder_path2, filename)

        # check if the folders exist, if not make them
        if not os.path.exists(folder_path1):
            os.makedirs(folder_path1)
        if not os.path.exists(folder_path2):
            os.makedirs(folder_path2)

        # check length of files in the folders
        num_files_in_ParamCombi1 = len(os.listdir("ParamCombi1"))
        num_files_in_ParamCombi2 = len(os.listdir("ParamCombi2"))    

        # Check which folder to save the file in depending on the number of files in the folders
        if num_files_in_ParamCombi1 >= num_files_in_ParamCombi2:
            if os.path.exists(file_path1): # is mse plot in folder 1?
                folder_name = folder_path2 # take the other folder for saving
            else:
                folder_name = folder_path1 # save in folder 1 if mse plot is not there
        elif not os.path.exists(file_path2): # if mse plot is not in folder 1 and not in folder 2?
                folder_name = folder_path1 # save in folder 1
        else:
                print("Error: File already exists in both folders")

        # Save the figure
        plt.savefig(os.path.join(folder_name, 'MSE.png'))
        #plt.show()