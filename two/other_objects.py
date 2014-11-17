import numpy as np
import copy

class Environment:
	""" Grid object that stores agent locations and state """
	def __init__(self, grid_size, predator_location, prey_location):
		""" Initialize environment of given size and predator, prey locations """
		self.grid_size = grid_size
		#Create empty grid
		self.grid = [[ ' ' for i in range(0, grid_size[0])] for y in range(0, grid_size[1])]
		#Get indices of predator, prey locations
		i = predator_location[0]
		j = predator_location[1]
		k = prey_location[0]
		l = prey_location[1]
		#Set representations of agents in grid
		self.grid[i][j] = 'X'
		self.grid[k][l] = 'O'
		#Store predator, prey locations
		self.predator_location = predator_location
		self.prey_location = prey_location

	def print_grid(self):
		""" Print the environment """
		print "=========="
		for row in self.grid:
			print row
		print "=========="

	def place_object(self, grid_object, new_location):
		""" Place an object at a given location in the environment"""
		#If predator
		if(grid_object == 'predator'):
			#Set 'X' at new location in grid
			self.grid[new_location[0]][new_location[1]] = 'X'
			#Update predator location
			self.predator_location = new_location
		#If prey
		elif(grid_object == 'prey'):
			#Set 'O' at new location in grid
			self.grid[new_location[0]][new_location[1]] = 'O'
			#Update prey location
			self.prey_location = new_location

	def move_object(self, object_name, new_location):
		""" Move object from old to new location in the grid """
		#If predator
		if(object_name == 'predator'):
			#Get old location
			old_location = self.predator_location
			#Update predator location
			self.predator_location = new_location
			#Empty old location in grid
			self.grid[old_location[0]][old_location[1]] = ' '
			#Set new location in grid to 'X'
			self.grid[new_location[0]][new_location[1]] = 'X'
		#If prey
		elif(object_name == 'prey'):
			#Get old location
			old_location = self.prey_location
			#Update prey location
			self.prey_location = new_location
			#Empty old location in grid
			self.grid[old_location[0]][old_location[1]] = ' '
			#Set new location in grid to 'O'
			self.grid[new_location[0]][new_location[1]] = 'O'			

	def get_size(self):
		""" Return environment size"""
		return self.grid_size

	def get_state(self):
		""" Return the current state that the environment's in """
		distance_x = abs(self.predator_location[0] - self.prey_location[0])
		distance_y = abs(self.predator_location[1] - self.prey_location[1])
		return [distance_x, distance_y]

	def get_location(self, object_name):
		""" Retrieve location of agent in grid """
		#Check which agent, and return its location
		if(object_name == 'predator'):
			return self.predator_location
		elif(object_name == 'prey'):
			return self.prey_location

class Policy:
	""" Policy object that stores action values for each state """
	def __init__(self, grid_size, policy_grid=None, prey=False):
		""" Initialize policy object of certain grid_size, with optional initial policy_grid and True for a prey """
		#Set grid size
		self.grid_size = grid_size
		#If the agent is not a prey, set the policy to random
		if prey==False:
			self.policy = {'North':15, 'East':15, 'South':15, 'West':15, 'Wait':15}
		#If the agent is a prey, set the policy to the prey-policy (80% wait, 20% action)
		else:
			self.policy = {'North':0.05, 'East':0.05, 'South':0.05, 'West':0.05, 'Wait':0.8}
		#If there is an initial policy, store it
		if(policy_grid is not None):
			self.policy_grid = policy_grid
		#Otherwise, create a grid of grid_size and fill it with the policy
		else:
			self.policy_grid = [[self.policy for i in range(0, self.grid_size[1])] for j in range(0, self.grid_size[0])]
		self.actions = {'North': [-1,0], 'East': [0,1], 'South': [1,0], 'West': [0,-1], 'Wait': [0,0]}

	def get_policy(self, state):
		""" Return the policy dictionary for a state """
		x_distance = state[0]
		y_distance = state[1]
		return self.policy_grid[x_distance][y_distance]

	def get_action(self, state, restricted=None):
		""" Choose an action and turn it into a move """
		#If there are restricted actions, do a restricted action pick
		if restricted is not None:
			chosen_action = self.pick_action_restricted(state, restricted)
		#Otherwise, just choose an action
		else:
			chosen_action = self.pick_action(state)
		#Get the transformation corresponding to the chosen action
		chosen_move = self.actions[chosen_action]
		#Return the name and transformation of the selected action
		return chosen_move, chosen_action		

	def get_e_greedy_action(self, state, epsilon):
		""" Do action selection using epsilon-greedy """
		policy = self.get_policy(state)
		best_action_list = []
		#Get the maximum value in the policy
		max_value = policy[max(policy)]
		#For each action, check if their value is maximum
		for action in policy.iteritems():
			#If value is max, append to best_action_list
			if action[1] == max_value:
				best_action_list.append(action[0])
		#Pick an action from the best_action_list at random
		chosen_action = np.random.choice(best_action_list)
		chosen_move = self.actions[chosen_action]
		return chosen_move, chosen_action

	def pick_action(self, state):
		""" Use the probabilities in the policy to pick a move """
		#Retrieve the policy for the current state
		policy = self.get_policy(state)
		#Zip the policy into a tuple of names, and a tuple of values
		action_name, policy = zip(*policy.items())
		#Use np.random.choice to select actions according to probabilities
		choice_index = np.random.choice(list(action_name), 1, p=list(policy))[0]
		#Return name of action
		return choice_index	

	def pick_action_restricted(self, state, blocked_moves):
		""" Use the probabilities in the policy to pick a move but can not perform blocked move """
		#Make a deep copy of the policy to prevent accidental pops
		temp_policy = copy.deepcopy(self.get_policy(state))
		update_probability = 0
		#Sum the probabilities of all blocked moves
		for block in blocked_moves:
			update_probability += temp_policy[block]
			#Remove the blocked actions from the policy
			del temp_policy[block]
		#Zip the cleaned policy into a tuple of names, and a tuple of values
		action_name, policy = zip(*temp_policy.items())
		#Divide the summed probabilities of blocked actions
		added_probability = update_probability/float(len(blocked_moves))
		#and add to the other actions
		new_policy = new_list = [x+added_probability for x in list(policy)]
		#Use np.random.choice to select actions according to probabilities
		choice_index = np.random.choice(list(action_name), 1, new_policy)[0]
		#Return name of action
		return choice_index


