import numpy as np
import random
from   collections import namedtuple, deque

from model_prj1 import QNetwork
#from model_prj1 import QNetwork2

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE  = 32         # minibatch size .................try 32 was 64
GAMMA       = 0.99            # discount factor
TAU         = 1e-3              # for soft update of target parameters
LR          = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size   = state_size
        self.action_size  = action_size
        self.seed         = random.seed(seed)

        # Q-Networks
        self.qnetwork_local  = QNetwork(state_size, action_size, seed).to(device)# local Neural Net, 'Y^'
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)# Target Neural Net, 'Y'
        self.optimizer       = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)  # run the neural_net to get action_values
        self.qnetwork_local.train()#.......................9-Jun-21
        #self.qnetwork_local.eval()

        # Epsilon-greedy action selection Is this reversed 12-Jun-21??
        #  r = random.random()
        #  if r < esp:
        #        return np.argmax(action_values.cpu().data.numpy()
        #  else:
        #        return random.choice(np.arange(self.action_size))
        
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        # '.detach() Returns the underlying raw stream separated from the buffer.
        #
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        #   This: self.qnetwork_target(next_states) returns a pytorch tensor of actions
        #                batch of 64 states
        #   plus this: self.qnetwork_target(next_states).detach() .. detaches the stream 
        #               into a numpy array of list holding the actions
        #   plus this: self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        #              makes Q-targets_next a list of the max action-value of the 4-actions
        #   Note: max(1) is the dimeson and max(1)[0] is the 1st element
        #
        # Compute Q targets for current states 
        Q_targets      = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected     = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        ''' If you replace your target DNNs weights with the local DNNs weights completely after some time,
            the predicted target Q values will change dramatically. If you change them step by step,
            the predicted target Q values will always change just a little bit. I suppose this consistency
            is beneficial for training. In order to keep the target and the local DNN different enough,
            tau has to be small. I think a soft update of the target DNN with a tau of 1e-3 should be
            compareable to a complete update of the target DNN after 1000 steps. Nevertheless, tau is a hyperparameter
            which can be tuned.
        '''

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory      = deque(maxlen=buffer_size)  
        self.batch_size  = batch_size
        self.experience  = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed        = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states  = torch.from_numpy(np.vstack([e.state  for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

