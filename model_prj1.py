import torch
import torch.nn as nn
import torch.nn.functional as F
drop_p = 0.2

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1  = nn.Linear(state_size, fc1_units)
        self.fc2  = nn.Linear(fc1_units, fc2_units)
        self.fc3  = nn.Linear(fc2_units, action_size)
        # Batch normaliztion: accelerates learning
        #
        # This removes the phenomenon of internal covariate shift, and addresses the problem by 
        # normalizing layer inputs.
        # The method draws its strength from making normalization a part of the model architecture and
        # performing the normalization for each training mini-batch. Batch Normalization allows one
        # to use much higher learning rates and be less careful about initialization. It also acts as a 
        # regularizer, in some cases eliminating the need  for Dropout. 
        #
        self.batch_norm1 = nn.BatchNorm1d(num_features=fc1_units)
        self.batch_norm2 = nn.BatchNorm1d(num_features=fc2_units)
    
    def forward(self, state):
        """Build a network that maps state -> action values."""
        #
        # Fully connected layers 1-3 
        #
        x = self.batch_norm1(F.relu(self.fc1(state))  )
        x = self.batch_norm2(F.relu(self.fc2(x))  )
        return self.fc3(x)
