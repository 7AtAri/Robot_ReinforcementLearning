### ###############################################################################################################
### Reinforcement Learning - WS 2023/24 
### 6-joint robot arm
### Group: Dennis Huff, Philipp Bodemann, Ari Wahl 
### ###############################################################################################################


import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque # queue for efficiently adding and removing elements from both ends
import gymnasium as gym
from gymnasium.envs.registration import register



class QNetwork(nn.Module):
    """ neural network that approximates the Q-value function: 

        inputs: state (concatenaded voxel_space and tcp_position) in the state space
        outputs: action values for each action in the action space
    """
    def __init__(self, state_size, action_size, hidden_size=101):
        super(QNetwork, self).__init__()
        print("State size:", state_size)
        print("Action size:", action_size)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(state_size, hidden_size)  # input layer to hidden layer
        self.relu = nn.ReLU()                          # activation function
        self.fc2 = nn.Linear(hidden_size, action_size) # hidden layer to output layer
        # Zugriff auf die Gewichtsmatrizen der ersten und zweiten Schicht
        fc1_weight_shape = self.fc1.weight.shape
        fc2_weight_shape = self.fc2.weight.shape

        print("Shape of fc1 weight matrix:", fc1_weight_shape)
        print("Shape of fc2 weight matrix:", fc2_weight_shape)
    
    def forward(self, state):
        x= self.flatten(state) # flattened input
        x= self.fc1(x) # 1st layer
        x = self.relu(x)  # activated first layer
        return self.fc2(x)    # output layer


class NStepReplayMemory:
    """ replay memory class stores n-step transitions
        and samples a random batch of transitions for training
        default: 3 step bootstrapping, discount factor 0.99
    """
    def __init__(self, capacity, n_steps=3, gamma=0.99):  # 3-step bootstrapping,  discount factor  0.99
        self.capacity = capacity          # max number of transitions that can be stored
        self.memory = deque(maxlen=capacity)   # deque for storing transitions
        self.n_steps = n_steps          # number of steps for bootstrapping
        self.gamma = gamma             # discount factor for bootstrapping
        self.n_step_buffer = deque(maxlen=n_steps) # deque for storing n-step transitions
     
    def push(self, state, tcp_position, action, reward, next_state, next_tcp_position, terminated, truncated): # stores a transition in the replay memory
        self.n_step_buffer.append((state, tcp_position, action, reward, next_state, next_tcp_position, terminated, truncated))  # stores experiences in a temporary buffer until n experiences are collected
        #print("next_state 1",next_state)
        if len(self.n_step_buffer) == self.n_steps: # if the buffer is full:
            R = sum(self.gamma ** i * reward for i, (_, _, _, reward, _, _, _, _) in enumerate(self.n_step_buffer)) # calculate the n-step return
            state, tcp_position, action, _, _, _, _, _ = self.n_step_buffer[0]            # get the first transition to get the state and action
            _, _, _, _, next_state, next_tcp_position, terminated, truncated = self.n_step_buffer[-1]    # get the last transition to get the next state and done
            self.memory.append((state, tcp_position, action, R, next_state, next_tcp_position, terminated, truncated)) #add the n-step transition to the memory
            if terminated or truncated: # if the episode is terminated or truncated, clear the buffer
                self.n_step_buffer.clear()
    
    def sample(self, sample_size): 
        return random.sample(self.memory, sample_size) # returns list of random samples from the memory
    
    def __len__(self): # returns current length of the memory
        return len(self.memory)
    

##### DQNAgent class ##############################################################################################
#  This class defines a DQNAgent that uses n-step bootstrapped returns and a target network to stabilize the learning process.                        #
# - act method:  chooses actions based on the current policy with an epsilon-greedy approach                                                                            #
# - learn method: 1) updates the online network using experiences sampled from memory, calculating n-step returns as targets.                       #
#                           2) target network's weights are synchronized with the online network at intervals defined by sync_rate,                                  # 
#                               (This ensures that the target values used in updates are stable but also gradually adapt to the improved policy.                 #
#                               The policy improvement in the provided DQNAgent code occurs implicitly through the updates made to the Q-network.     #
#                               The action with the highest Q-value in a given state is considered the best action according to the current policy)              #
##################################################################################################################
    
class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, lr=0.001, memory_size=10000, sample_size=64, n_steps=3, sync_rate=1000):
        # init parameters for the DQN Agent
        self.state_size = state_size                  # state space size
        self.action_size = action_size              # action space size
        self.gamma = gamma ** n_steps         # discount factor raised to the power of n_steps ---> n-step returns
        self.batch_size = sample_size              # number of experiences to sample from memory during learning
        self.sync_rate = sync_rate                   # how often to sync the target network with the online network
        self.steps_done = 0                              # total number of steps (counter)
        
        # initialize replay memory with specified size and n-step setting
        self.memory = NStepReplayMemory(memory_size, n_steps, gamma)
        # initialize  the online (primary) Q-network
        self.model = QNetwork(state_size, action_size).to(device)
        # initialize the target Q-network + load it with weights from online network
        self.target_model = QNetwork(state_size, action_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()  # set target network to eval mode
        # set optimizer for updating the online network
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    

    # todo: eventually rather implement epsilon decay
    def act(self, state, epsilon=0.1): 
        # select action according to epsilon-greedy policy
        if random.random() > epsilon: 
            # (1 - epsilon)-probability, choose the best action based on the current policy --> exploit
            state = torch.FloatTensor(state).unsqueeze(0).to(device)  # state needs to be in tensor and add batch dimension
            with torch.no_grad():  # gradient calculation needs to be disabled for inference
                q_values = self.model(state)  # get q-values for all actions
            return np.argmax(q_values.cpu().numpy())  # return action with highest q-value (best action)
        else:
            # with probability epsilon, choose a random action --> explore
            return random.randrange(self.action_size)
    

    def learn(self): 
        # Online-Netzwerkupdate basierend auf Anzahl der Erfahrungen
        if len(self.memory) < self.batch_size:
            return  # Kein Lernen, wenn nicht genügend Samples im Speicher vorhanden sind

        # Abtasten von Erfahrungen aus dem Speicher:
        transition_samples = self.memory.sample(self.batch_size)
        training_batch = list(zip(*transition_samples))  

        # Konvertieren von Batches in Tensoren:
        states, actions, rewards, next_states, dones = map(torch.FloatTensor, training_batch)
        actions = actions.long()  
        dones = dones.float() 
        states = torch.tensor(states, device=device, dtype=torch.float32)
        actions = torch.tensor(actions, device=device, dtype=torch.long)
        rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
        next_states = torch.tensor(next_states, device=device, dtype=torch.float32)
        dones = torch.tensor(dones, device=device, dtype=torch.float32)

        # Vorwärtsdurchlauf durch das Netzwerk
        if len(states) != 6161:  # Überprüfen Sie die Größe des Zustandsvektors
            print("Fehler: Der Zustandsvektor hat nicht die richtige Größe.")
            return  # Beenden Sie den Vorwärtsdurchlauf, wenn der Zustandsvektor nicht korrekt ist
        else:
            state_action_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_state_values = self.target_model(next_states).max(1)[0].detach()
            expected_state_action_values = rewards + (self.gamma * next_state_values * (1 - dones))
            loss = nn.MSELoss()(state_action_values, expected_state_action_values)

            # Gradientenabstieg zur Minimierung des Verlusts
            self.optimizer.zero_grad()  
            loss.backward()  
            self.optimizer.step()  

            # Periodisches Aktualisieren der Target-Netzwerk-Gewichte
            if self.steps_done % self.sync_rate == 0: 
                self.target_model.load_state_dict(self.model.state_dict())

            self.steps_done += 1


    # def learn(self): #, episodes, render=False, is_slippery=False)
    #     # online network update based on number of experiences
    #     if len(self.memory) < self.batch_size:
    #         return  # no learning if not enough samples in memory
        
    #     # sample experiences from memory:
    #     transition_samples = self.memory.sample(self.batch_size)
    #     training_batch = list(zip(*transition_samples))  # convert the batch to separate states, actions, etc.
    #     # zip function combines elements from each of the input iterables (lists, tuples, etc.) based on their position. 
    #     # * operator, when used with an iterable (like "transition_samples"), unpacks the iterable 
    #     # ---> each element goes as a separate argument to the zip
    #     # then zip is called and it combines the first elements of each tuple (all states),
    #     # the second elements (all actions), etc., into new tuples.

    #     # convert batches to tensors:
    #     states, actions, rewards, next_states, dones = map(torch.FloatTensor, training_batch)
    #     print("netx_state",next_states)
    #     actions = actions.long()  # actions should be long tensors
    #     dones = dones.float() # terminated states are set to float for further calculations
    #     states = torch.tensor(states, device=device, dtype=torch.float32)
    #     actions = torch.tensor(actions, device=device, dtype=torch.long)
    #     rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
    #     next_states = torch.tensor(next_states, device=device, dtype=torch.float32)
    #     dones = torch.tensor(dones, device=device, dtype=torch.float32)

    #     # calc q-values for selected actions
    #     state_action_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    #     # get the next state values from target network 
    #     next_state_values = self.target_model(next_states).max(1)[0].detach()
    #     # get expected q-values (target values) using n-step returns
    #     expected_state_action_values = rewards + (self.gamma * next_state_values * (1 - dones))
    #     # calculates the expected Q-values for each state-action pair, and excludes the value of future states when done is True for a given state.

    #     # calculate loss between the expected q-values and the ones predicted by the online network
    #     loss = nn.MSELoss()(state_action_values, expected_state_action_values)
        
    #     # perform gradient descent to minimize the loss
    #     self.optimizer.zero_grad()  # reset gradients to zero
    #     loss.backward()  # compute gradient of the loss 
    #     self.optimizer.step()  # update network weights
        
    #     # periodically update target network weights
    #     if self.steps_done % self.sync_rate == 0: # if the number of steps is a multiple of the sync rate
    #         self.target_model.load_state_dict(self.model.state_dict()) # sync weights from online network to target network
        
    #     self.steps_done += 1  # raise total number of steps by one




if __name__ == "__main__":
    # check which device is available
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    register(
        id='RobotEnvironment-v0',
        entry_point='Environment:RobotEnvironment',
    )

    env = gym.make('RobotEnvironment-v0')
   
    # get env specs to init the agent
    state_size = env.observation_space.shape[0]
    print(state_size)
    #action_size = env.action_space.n
    action_size = env.action_space.nvec.prod()
    
    agent = DQNAgent(state_size, action_size)
    num_episodes = 30 # start with low number and raise ...of use as hyperparameter
    visualization_rate = 10 # how often to visualize the environment
    terminated = False # True if the agent leaves the trajectory of the helix

    for episode in range(num_episodes):
        voxel_space, tcp_position, info = env.reset()  # reset the environment
        
        print("Observation after reset:", voxel_space)
        
        state = voxel_space
        actual_size = state.size
        #state = env.reset()[0]  # reset the environment to state 0
        #state = np.reshape(state, [1, state_size])  # reshape state for compatibility with DQNAgent
        
        # Vor dem Umformen des Zustands, überprüfe die Form des Arrays
        print("Zustandsform vor der Umformung:", state.shape)

        # Stelle sicher, dass die Größe des Zustandsarrays mit der erwarteten Größe übereinstimmt
        expected_size =   61  * 61 * 101 # Erwartete Größe des Zustandsarrays

        # Führe die Umformung nur durch, wenn die Größe des Zustandsarrays korrekt ist
        if actual_size == expected_size:
            state = np.reshape(state, [1, expected_size])
            print("Zustand erfolgreich umgeformt:", state.shape)
        else:
            print("Fehler: Die Größe des Zustandsarrays stimmt nicht mit der erwarteten Größe überein.")

        total_reward = 0
        rewards_per_episode = []
        #epsilon_history = []

        while not terminated:
            action = agent.act(state)  # select action according to epsilon-greedy policy
            next_state, reward, terminated , _ = env.step(action)  # take action in environment
            next_state = np.reshape(next_state, [1, state_size])
            
            # store transition in memory
            agent.memory.push(state, action, reward, next_state, terminated)
            
            state = next_state  # move to the next state
            total_reward += reward
            if reward == 1:
                rewards_per_episode.append(1)

                # evlt break oder if abfrage vor dem lernen (if not termqinated = True ???)
                agent.learn()  # learn from memory
            
            # visualize the environment regularly
            if episode % visualization_rate == 0:
                env.render()   
                #pass

        print(f"Episode: {episode+1}, Total reward: {total_reward}")

    env.close()  # close the environment



## todo: maybe define a train() and test() function ...