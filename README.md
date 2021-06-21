
# Udacity RL Navigation Project #1

The contents of this ‘Readme.md’ file describes the reinforced deep learning project number one of three. The solution reused the Udacity lesson code regarding Double Queue Learning. This solution uses a deep double queue network neural architecture. Files from the lesson have been modified and they are the dqn_agent and model ‘.py’ files stored in the ‘Main Git Branch’.

For this project the objective is to train an agent to navigate (and collect bananas!) in a large, square world.
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. The goal of  the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space/size has 37 dimensions offered by the environment and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

•	0 - move forward.
•	1 - move backward.
•	2 - turn left.
•	3 - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.
This solution achieved +13 results at Environment solved in 293 episodes with an average Score of 
13.07.

# Learning Algorithm
The algorithm applied is the double Q_network where the current nn-network or local-network (i.e., Y-Hat prediction) is one network and the 'labeled-data "Y"'
or target-network is the other neural network. Driven by the double_dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995) procedure; the core algorithm uses the current state and calls act.env(current - state) process which trains the local-network, then applies Epsilon-Greedy to return the 'best' action which is then used to get the next-state, rewards, and done status.
A step is now taken which saves current state and next-state plus action, reward and done to reply-memory. If the reply-memory has enough experiences accumulated then a random sample of experiences are selected and these are used to train the local-network; otherwise the reply-memory continues to build. Note, the random sample returned are the following elements: states, actions, rewards, next-states, dones.  The next-state's targets = rewards + (gamma * Q_targets_next * (1 - dones)) where Q_targets are a list of the max action-values of the 4-actions available for that next-state; correspondingly the local network's actions are gathered as expected values. Both of these values go into the loss function F.mse_loss(expected, targets) followed by the weight optimization phase then the process iterates. Every four steps the target-network is updated with the local-network's weights. The reply-memory grows with experiences and these are used to lean.

# Environment
The code we executed in the Udacity provided jupyter workspace, the specifics follow:
1.	Ipython 6.5.0 imported to the workspace: !pip -q install ./python
2.	The code is written in PyTorch 0.4.0 (provided by Udacity  environment) and Python 3.6 
3.	Workspace URL: Home (udacity-student-workspaces.com)
4.	The code was run using a ‘CPU’ based environment and not ‘GPU’. 
## Core files stored in the environment are:
    Dqn_agent_prj1.py
    Model_prj1.py
    Trained_model.pt
    Navigatio.jpynb

# Environment Set Up
Within the Udacity workspace the following installation procedures were executed:
1.	!pip -q install ./python;   Install Python
2.	from unityagents import UnityEnvironment
3.	Import numpy as np
4.	Set the  banana environment within the workspace:
5.	    env = UnityEnvironment(file_name="/data/Banana_Linux_NoVis/Banana.x86_64")
6.	Import all necessary packages:
7.	  import random
8.	  import torch
9.	  from collections import deque
10.	  import matplotlib.pyplot as plt
11.	  %matplotlib inline
12.	  !python -m pip install pyvirtualdisplay
13.	  from pyvirtualdisplay import Display
14.	  display = Display(visible=0, size=(1400, 900))
15.	  display.start()
16.	  is_ipython = 'inline' in plt.get_backend()
17.	  if is_ipython:
18.	    from IPython import display
19.	    plt.ion()
20.	  import os
21.	  from pathlib import Path
22.	  from dqn_agent_prj1 import Agent
23.	# Define two procedures:
24.	  def get_ns_rewrd_done(env_info): returns next_state, reward, done
25.	  def double_dqn(n_episodes=2000,
26.	                 max_t=1000, eps_start=1.0, eps_end=0.01, 
27.	                 eps_decay=0.995) ; this is the main procedure to train the agent.
# Train the Agent / Execution
Within the workspace execute the following command sequence:
1.	Set up the environment as explained above
2.	Run: env_info = env.reset(train_mode=True)[brain_name]
7.	Set filename;  file_name = dir + '/trained_model.pt'; sets the saved trained model
3.	Set the hyper-parameters:
4.	    BATCH_SIZE  = 32
5.	    LR          = 5e-55 
6.	    GAMMA       = 0.95
7.	    eps_decay   =  0.9645
8.	    agent       = Agent(state_size=37, action_size=4, seed=42)
9.	    
10.	# Initialize global parameters:
11.	scores      = 0
12.	# Run the program:
13.	print('BATCH_SIZE',BATCH_SIZE )
14.	scores    = double_dqn(n_episodes=700, max_t=1000,
                        eps_start  =  1.0,
                        eps_end   =  0.01,
                        eps_decay= 0.9645)
    fig       = plt.figure()
    ax        = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
# Best Training Results

The environment is considered solved when the agent’s training score is above 13 and the number of episodes is over 100.
Below the hyper-parameters setting yielded the following output:
## Results:
    BATCH_SIZE 32
    Episode 100	Average Score: 2.33
    Episode 200	Average Score: 7.11
    Episode 300	Average Score: 9.86
    Episode 393	Average Score: 13.07
    Environment solved in 293 episodes!	Average Score: 13.07

![image](https://user-images.githubusercontent.com/86236466/122802190-10dee280-d293-11eb-8757-475415e43772.png)


