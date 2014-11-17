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

	def get_action(self, state, restricted=None, epsilon=0.0, discount_factor=0.0, alpha=0.0, predator=True):
		""" Choose an action and turn it into a move """
		chosen_action = self.pick_action(state, restricted=restricted, epsilon=epsilon)
		#Get the transformation corresponding to the chosen action
		chosen_move = self.actions[chosen_action]
		return chosen_move, chosen_action		

	def q_learning(self, old_state, action, new_state, epsilon, discount_factor, alpha, reward):
		#Store value of current state-action pair
		q_value = self.get_policy(old_state)[action]
		print "Old Q-value for state ", old_state, " action, ", action, ":", q_value
		if(reward == True):
			score = 10
		else:
			score = 0
		#Get value of max action of the new state
		chosen_move, chosen_action = self.get_action(new_state)
		next_q_value = self.get_policy(new_state)[chosen_action]
		new_q_value = q_value + alpha * (score + discount_factor * next_q_value - q_value)
		self.get_policy(old_state)[action] = new_q_value
		print self.policy_grid[old_state[0]][old_state[1]][action]

	def get_e_greedy_policy(self, policy, epsilon=0.0):
		#Get |A(s)|
		number_actions = len(policy)
		#Get the extra probability to divide over actions
		extra_probability = epsilon/number_actions
		best_action_list = []
		other_action_list = []
		#Get the maximum value in the policy
		max_value = policy[max(policy)]
		#For each action, check if their value is maximum
		for action in policy.iteritems():
			#If value is max, append to best_action_list
			if action[1] == max_value:
				best_action_list.append(action[0])
			#Otherwise, append to other_action_list
			else:
				other_action_list.append(action[0])
		probability_dict = {}
		#Compute the probability of the best actions
		best_actions_probability = (1.0 - epsilon)/len(best_action_list)
		#The best actions have a probability of best_actions_probability + extra_probability
		for max_action in best_action_list:
			probability_dict[max_action] = best_actions_probability + extra_probability
		#The other actions have a probability of extra_probability
		for other_action in other_action_list:
			probability_dict[other_action] = extra_probability
		return probability_dict


	def pick_action(self, state, restricted=None, epsilon=0.0):
		""" Use the probabilities in the policy to pick a move """
		#Retrieve the policy using e_greedy for the current state
		policy = self.get_e_greedy_policy(self.get_policy(state), epsilon)
		if(restricted is not None):
			#make a deepcopy of the policy to prevent accidental pops
			temp_policy = copy.deepcopy(policy)
			update_probability = 0
			#Sum the probabilities of all blocked moves
			for block in restricted:
				update_probability += temp_policy[block]
				#Remove the blocked actions from the policy
				del temp_policy[block]
			#Zip the policy into a tuple of names, and a tuple of values
			action_name, policy = zip(*temp_policy.items())
			#Divide the summed probabilities of blocked actions
			added_probability = update_probability/float(len(self.get_policy(state))-len(restricted))
			#and add to the other actions
			policy = [x+added_probability for x in list(policy)]			
		else:	
			#Zip the policy into a tuple of names, and a tuple of values
			action_name, policy = zip(*policy.items())			
		#Use np.random.choice to select actions according to probabilities
		choice_index = np.random.choice(list(action_name), 1, p=list(policy))[0]
		#Return name of action
		return choice_index	
