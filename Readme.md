[//]: # (Image References)

[image1]: https://wpumacay.github.io/research_blog/imgs/gif_banana_agent.gif "Trained Banana Agents"

![Trained Banana Agents][image1]

# Udacity RL Navigation Project #1

The contents of this ‘Readme.md’ file describes the reinforced deep learning project number one of three. The solution reused the Udacity lesson code regarding Double Queue Learning. 
This solution uses a deep double queue network neural architecture. Files from the lesson have been modified and they are the dqn_agent and model ‘.py’ files stored in the ‘Main Git Branch’.

For this project the objective is to train an agent to navigate (and collect bananas!) in a large, square world.
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. The goal of  the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space/size has 37 dimensions offered by the environment and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information,
 the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:
-	0 - move forward
-	1 - move backward
-	2 - turn left
-	3 - turn right


The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.
This solution achieved +13 results at Environment solved in 293 episodes with an average Score of 
13.07.

# Learning Algorithm
 
The algorithm applied is the double Q-network where the current nn-network or local-network (i.e., Y-Hat prediction) is one network and the 'labeled-data "Y"'
or target-network is the other neural network. Driven by the double-dqn(n-episodes=2000, max-t=1000, eps-start=1.0, eps-end=0.01, eps-decay=0.995) procedure;
the core algorithm uses the current state and calls act.env(current - state) process which trains the local-network, then applies Epsilon-Greedy to return the 'best' action which is then used to get the next-state, rewards,
and done status.  A step is now taken which saves current state and next-state plus action, reward and done to reply-memory. If the reply-memory has enough experiences accumulated then a random sample of 
experiences are selected and these are used to train the local-network; otherwise the reply-memory continues to build.
Note, the random sample returned are the following elements: states, actions, rewards, next-states, dones.  The next-state's targets = rewards + (gamma * Q-targets_next * (1 - dones))
where Q-targets are a list of the max action-values of the 4-actions available for that next-state; correspondingly the local network's actions are gathered as expected values.
Both of these values go into the loss function F.mse-loss(expected, targets) followed by the weight optimization phase then the process iterates. Every four steps the target-network is updated with the local-network's weights.
The reply-memory grows with experiences and these are used to lean.

# Environment

The code we executed in the Udacity provided jupyter workspace, the specifics follow:
1.	Ipython 6.5.0 imported to the workspace: !pip -q install ./python
2.	The code is written in PyTorch 0.4.0 (provided by Udacity  environment) and Python 3.6 
3.	Workspace URL: Home (udacity-student-workspaces.com)
4.	The code was run using a ‘CPU’ based environment and not ‘GPU’. 
## Core files stored in the environment are:
-    'Dqn_agent-prj1.py'
-    'Model_prj1.py'
-    'Trained_model.pt'
-    'Navigatio.jpynb'


## Down load the environment

Four core elements are needed to run this prject:

1.	UDACITY provided Workspace
2.	Python 3.x 
3.	The non-visual Unity Environment
4.	The  project solution files mentioned above


The following text descibes how to download the Environment so that the projecct solution can be executed.

From Project instructions I choose to use the course's provided Workspace for this project, I did not run the
project on my PC. To download the project to a PC review the UDACITY material at this URL: 

https://github.com/udacity/deep-reinforcement-learning#dependencies

Since the projects is utilizing the UDACITY Workspace,  the Unity ML agent visual simulator cannot be run within the class provided
environemnt; instead the non visual part of this Unity environment is used to train the agent. The UDACITY Workspace provides
a Jupyter server that is directly in my browser with GPU or CPU support; Note for this exercise I did not use GPU.
As one enters the Workspace a choice of GPU or CPU is available for the operator to select. The UnityEnvironment environment is provided
by UDACITY and it is already saved in the Workspace; it can be accessed at the file path "/data/Banana_Linux_NoVis/Banana.x86_64".
The following instructions describe the steps neccessary to download and prepare the Workspace envrionment so that the project
code can be run.

###Step 1: 
	Enter the workspace provided by UDACITY by selecting the Projects Workspace option. The workspace will launch and query for GPU or CPU, select CPU.

###Step 2:  Load Python 3.6 into the environemnt
	Run the next code cell to install a few packages. This line will take a few minutes to run!
	Within the Jupyter Cell execute this instruction to load Python:
		 [1] !pip -q install ./python

		Output Message follows:
		tensorflow 1.7.1 has requirement numpy>=1.13.3, but you'll have numpy 1.12.1 which is incompatible.
		python 6.5.0 has requirement prompt-toolkit<2.0.0,>=1.0.15, but you'll have prompt-toolkit 3.0.19 which is incompatible.

###Step 3: Load the Unity non-visual environment into the work space
	The environment is already saved in the Workspace; run the next code cell to install the Unity Envrionment.
  		[2] from unityagents import UnityEnvironment
		     import numpy as np
		      '# please do not modify the line below'
		       env = UnityEnvironment(file_name="/data/Banana_Linux_NoVis/Banana.x86_64")

		Output Message follows:
		INFO:unityagents:
		'Academy' started successfully!
		Unity Academy name: Academy
			Number of Brains: 1
			Number of External Brains : 1
			Lesson number : 0
			Reset Parameters :
		
		Unity brain name: BananaBrain
		        Number of Visual Observations (per agent): 0
		        Vector Observation space type: continuous
		        Vector Observation space size (per agent): 37
		        Number of stacked Vector Observation: 1
		        Vector Action space type: discrete
		        Vector Action space size (per agent): 4 
		        Vector Action descriptions: , , , 

###Step 4:   Environments contain brains which are responsible for deciding the actions of their associated agents.
	Check for the first brain available, and set it as the default brain we will be controlling from Python.
	Run the next code cell to set the brain variables.

		[3] '# get the default brain'
		      brain_name = env.brain_names[0]
		      brain      = env.brains[brain_name]

		Examine the State and Action Spaces
		Run the code cell below to print some information about the environment.
	
		[4] '# reset the environment'
		env_info = env.reset(train_mode=True)[brain_name]
		
		'# number of agents in the environment'
		print('Number of agents:', len(env_info.agents))
		
		'# number of actions'
		action_size = brain.vector_action_space_size
		print('Number of actions:', action_size)
		
		# examine the state space 
		state = env_info.vector_observations[0]
		print('States look like:', state)
		state_size = len(state)
		print('States have length:', state_size)

		Output Message follows:
		Number of agents: 1
		Number of actions: 4
		States look like: [ 1.          0.          0.          0.          0.84408134  0.          0.
		  1.          0.          0.0748472   0.          1.          0.          0.
		  0.25755     1.          0.          0.          0.          0.74177343
		  0.          1.          0.          0.          0.25854847  0.          0.
		  1.          0.          0.09355672  0.          1.          0.          0.
		  0.31969345  0.          0.        ]
		States have length: 37

###Step 5:   The proper environments are set up for Python and Unity so now we take Random Actions in the Environment and test.
	Note that in this coding environment, you will not be able to watch the agent while it is training, 
	and you should set train_mode=True to restart the environment.

	Run the code cell below
	 [5]  env_info = env.reset(train_mode=True)[brain_name] # reset the environment
	state = env_info.vector_observations[0]            # get the current state
	score = 0                                          # initialize the score
	while True:
		action     = np.random.randint(action_size)        # select an action
		env_info   = env.step(action)[brain_name]        # send the action to the environment
		next_state = env_info.vector_observations[0]   # get the next state
		reward = env_info.rewards[0]                   # get the reward
		done   = env_info.local_done[0]                  # see if episode has finished
		score += reward                                # update the score
		state  = next_state                             # roll over the state to next time step
		if done:                                       # exit loop if episode finished
			break
			print("Score: {}".format(score))

	Output Message follows:
	Score: 0.0

## Prepare the Jupyter Workspace to Run Training

###Step 1:   The proper environments have now been down loaded and tested. This step will load the solution code and libraries
	Run the next code cells  to define the python procedures.
	First set up the file-name for the final trained model
	[6] '#   Establish path to files'
		import os
		dir = os.getcwd()
		file_name = dir + '/trained_model.pt'

###Step 2:Import the libraries and set  up Jupyter display 
	Run the next code cells  to define the python procedures.
	[7] '# '
	'#. Import the Necessary Packages'
	'#'
	'import random'
	'import torch'
	'import numpy as np'
	'from collections import deque'
	'import matplotlib.pyplot as plt'
	'%matplotlib inline'
	!python -m pip install pyvirtualdisplay
	from pyvirtualdisplay import Display
	display = Display(visible=0, size=(1400, 900))
	display.start()

	[8] '#  Run code for display '

	is_ipython = 'inline' in plt.get_backend()
	if is_ipython:
	    from IPython import display
	plt.ion()

	Output Message follows:
		Collecting pyvirtualdisplay
		Downloading https://files.pythonhosted.org/packages/79/30/e99e0c480a858410757e7516958e149285ea08ed6c9cfe201ed0aa12cee2/PyVirtualDisplay-2.2-py3-none-any.whl
		Collecting EasyProcess (from pyvirtualdisplay)
		Downloading https://files.pythonhosted.org/packages/48/3c/75573613641c90c6d094059ac28adb748560d99bd27ee6f80cce398f404e/EasyProcess-0.3-py2.py3-none-any.whl
		Installing collected packages: EasyProcess, pyvirtualdisplay
		Successfully installed EasyProcess-0.3 pyvirtualdisplay-2.2

###Step 3:   Define the three procedures and import the model class and agent
	Run the next code cells  to define the python procedures.

	[9] # Define two procedures:
	def get_ns_rewrd_done(env_info): returns next_state, reward, done
	def double_dqn(n_episodes=2000,max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995) ; this is the main procedure to train the agent.
	
	[10] # Load the agent: with the specifed parameters
	from dqn_agent_prj1 import Agent
	agent = Agent(state_size=37, action_size=4, seed=42)

	[11] #  Define the main procedure for the double DQN using the specified parameters
	def double_dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):


## Train the Agent / Execution
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


