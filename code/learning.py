import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque # queue for efficiently adding and removing elements from both ends
import gym



class QNetwork(nn.Module):
    """ neural network that approximates the Q-value function: 

        inputs: states in the state space
        outputs: action values for each action in the action space
    """
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)  # input layer to hidden layer
        self.relu = nn.ReLU()                          # activation function
        self.fc2 = nn.Linear(hidden_size, action_size) # hidden layer to output layer
    
    def forward(self, state):
        x= self.fc1(state) # 1st layer
        x = self.relu(x)  # activated first layer
        return self.fc2(x)            # output layer


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
     
    def push(self, state, action, reward, next_state, done): # stores a transition in the replay memory
        self.n_step_buffer.append((state, action, reward, next_state, done))  # stores experiences in a temporary buffer until n experiences are collected
        if len(self.n_step_buffer) == self.n_steps: # if the buffer is full:
            R = sum(self.gamma ** i * reward for i, (_, _, reward, _, _) in enumerate(self.n_step_buffer)) # calculate the n-step return
            state, action, _, _, _ = self.n_step_buffer[0]            # get the first transition to get the state and action
            _, _, _, next_state, done = self.n_step_buffer[-1]    # get the last transition to get the next state and done
            self.memory.append((state, action, R, next_state, done)) #add the n-step transition to the memory
            if done:
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
        self.model = QNetwork(state_size, action_size)
        # initialize the target Q-network + load it with weights from online network
        self.target_model = QNetwork(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()  # set target network to eval mode
        # set optimizer for updating the online network
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    
    def act(self, state, epsilon=0.1):
        # select action according to epsilon-greedy policy
        if random.random() > epsilon: 
            # (1 - epsilon)-probability, choose the best action based on the current policy --> exploit
            state = torch.FloatTensor(state).unsqueeze(0)  # state needs to be in tensor and add batch dimension
            with torch.no_grad():  # gradient calculation needs to be disabled for inference
                q_values = self.model(state)  # get q-values for all actions
            return np.argmax(q_values.numpy())  # return action with highest q-value (best action)
        else:
            # with probability epsilon, choose a random action --> explore
            return random.randrange(self.action_size)
    
    def learn(self) #, episodes, render=False, is_slippery=False):
        # online network update based on number of experiences
        if len(self.memory) < self.batch_size:
            return  # no learning if not enough samples in memory
        
        # sample experiences from memory:
        transition_samples = self.memory.sample(self.batch_size)
        training_batch = list(zip(*transition_samples))  # convert the batch to separate states, actions, etc.
        # zip function combines elements from each of the input iterables (lists, tuples, etc.) based on their position. 
        # * operator, when used with an iterable (like "transition_samples"), unpacks the iterable 
        # ---> each element goes as a separate argument to the zip
        # then zip is called and it combines the first elements of each tuple (all states),
        # the second elements (all actions), etc., into new tuples.

        # convert batches to tensors:
        states, actions, rewards, next_states, dones = map(torch.FloatTensor, training_batch)
        
        # calc q-values for selected actions
        state_action_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # get the next state values from target network 
        next_state_values = self.target_model(next_states).max(1)[0].detach()
        # get expected q-values (target values) using n-step returns
        expected_state_action_values = rewards + (self.gamma * next_state_values * (1 - dones))
        
        # calculate loss between the expected q-values and the ones predicted by the online network
        loss = nn.MSELoss()(state_action_values, expected_state_action_values)
        
        # perform gradient descent to minimize the loss
        self.optimizer.zero_grad()  # reset gradients to zero
        loss.backward()  # compute gradient of the loss 
        self.optimizer.step()  # update network weights
        
        # periodically update target network weights
        if self.steps_done % self.sync_rate == 0: # if the number of steps is a multiple of the sync rate
            self.target_model.load_state_dict(self.model.state_dict()) # sync weights from online network to target network
        
        self.steps_done += 1  # raise total number of steps by one




if __name__ == "__main__":

    env = gym.make('environment.py') # create environment
    # get env specs to init the agent
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    num_episodes = 30 # start with low number and raise ...of use as hyperparameter
    visualization_rate = 10 # how often to visualize the environment

    for episode in range(num_episodes):
        state = env.reset()  # Reset the environment for a new episode
        state = np.reshape(state, [1, state_size])  # Reshape state for compatibility with DQNAgent
        
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)  # Select action according to policy
            next_state, reward, done, _ = env.step(action)  # Take action in environment
            next_state = np.reshape(next_state, [1, state_size])
            
            # Store the transition in replay memory
            agent.memory.push(state, action, reward, next_state, done)
            
            state = next_state  # Move to the next state
            total_reward += reward
            
            agent.learn()  # Learn from replay memory
            
        print(f"Episode: {episode+1}, Total reward: {total_reward}")
        
        # visualize the environment regularly
        if episode % visualization_rate == 0:
            env.render()