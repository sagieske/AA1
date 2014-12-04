import numpy as np
import copy
import itertools
import math
import operator
import sys
import operator
import random

class Environment:
	""" Grid object that stores agent locations and state """
	def __init__(self, grid_size, predator_location_list, prey_location):
		""" Initialize environment of given size and predator, prey locations """
		self.grid_size = grid_size
		#Create empty grid
		self.grid = [[ ' ' for i in range(0, grid_size[0])] for y in range(0, grid_size[1])]
		#Get indices of predator, prey locations
		i = predator_location_list[0][0]
		j = predator_location_list[0][1]
		k = prey_location[0]
		l = prey_location[1]
		#Set representations of agents in grid
		self.grid[i][j] = 'X'
		self.grid[k][l] = 'O'
		#Store predator, prey locations
		# Create and populate predator location list with locations
		self.predator_location_list = []
		for i in range(0, len(predator_location_list)):
                	self.predator_location_list.append(predator_location_list[i])
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

	def move_object(self, object_name, new_location_list):
		""" Move object from old to new location in the grid """
		#If predator
		if(object_name == 'predator'):
			#Get old location
			old_location_list = self.predator_location_list
			#Update predator location
			self.predator_location_list = new_location_list
			for i in range(0, len(old_location_list)):
				#Empty old location in grid
				self.grid[old_location_list[i][0]][old_location_list[i][1]] = ' '
				#Set new location in grid to 'X'
				self.grid[new_location_list[i][0]][new_location_list[i][1]] = 'X'
		#If prey
		elif(object_name == 'prey'):
			#print 'new location list: ', new_location_list 
			#Get old location
			old_location = self.prey_location
			#Update prey location
			self.prey_location = new_location_list
			#Empty old location in grid
			self.grid[old_location[0]][old_location[1]] = ' '
			#Set new location in grid to 'O'
			self.grid[new_location_list[0]][new_location_list[1]] = 'O'			

	def get_size(self):
		""" Return environment size"""
		return self.grid_size

	def get_state(self):
		""" Return the current state that the environment's in """
		return self.predator_location_list, self.prey_location #[self.predator_location[0], self.predator_location[1], self.prey_location[0], self.prey_location[1]]

	def get_location(self, object_name, index=0):
		""" Retrieve location of agent in grid """
		#Check which agent, and return its location
		if(object_name == 'predator'):
			return self.predator_location_list[index]
		elif(object_name == 'prey'):
			return self.prey_location

class Policy:
	""" Policy object that stores action values for each state """
	def __init__(self, grid_size, policy_grid=None, prey=False, softmax=False):
		""" Initialize policy object of certain grid_size, with optional initial policy_grid and True for a prey """
		#Store prey boolean
		self.prey = prey
		#Store on-policy MC boolean
		#Store grid size
		self.grid_size = grid_size
		self.multi_agents = True
		#If the agent is not a prey, set the policy to random with initialization value
		if prey==False:
			self.policy = {'North':0.2, 'East':0.2, 'South':0.2, 'West':0.2, 'Wait':0.2}
		#If the agent is a prey, set the policy to the prey-policy (80% wait, 20% action)
		else:
			self.policy = {'North':0.2, 'East':0.2, 'South':0.2, 'West':0.2, 'Wait':0.2}
		#If a policy_grid is given, store it
		if(policy_grid is not None):
			self.policy_grid = policy_grid
		else:
			self.policy_grid = [[[[copy.deepcopy(self.policy) for i in range(0, self.grid_size[1])] for j in range(0, self.grid_size[0])] for k in range(0, self.grid_size[1])] for l in range(0, self.grid_size[0])]	
		#Store the actions and their corresponding transformations
		self.actions = {'North': [-1,0], 'East': [0,1], 'South': [1,0], 'West': [0,-1], 'Wait': [0,0]}

		# Q-value table for possible distances from prey
		encoding = True
		if encoding:
			# calculate max distance and create dictionary with None values as Q values
			max_distance_x =  int(math.ceil(self.grid_size[0]/2.0))
			max_distance_y =  int(math.ceil(self.grid_size[1]/2.0))
			range_max_distance_x = range(0, max_distance_x)
			range_max_distance_y = range(0, max_distance_y)
			# Get all possible absolute distances from prey
			distances = list(itertools.product(range_max_distance_x , range_max_distance_y))
			self.distance_dict = dict.fromkeys(distances, 15)
		self.softmax = softmax

	def update_Q_values(self,Q, pair):
		state = pair[0]
		i = state[0]
		j = state[1]
		k = state[2]
		l = state[3]
		action = pair[1]
		self.policy_grid[i][j][k][l][action] = Q

	def get_policy(self, state):
		""" Return the policy dictionary for a state """
		if self.multi_agents == False:
			i = state[0]
			j = state[1]
			k = state[2]
			l = state[3]
			return self.policy_grid[i][j][k][l]
		else:
			return {'North': 0.2, 'East': 0.2, 'South':0.2, 'West':0.2, 'Wait':0.2} 	

	def get_action(self, state, epsilon=0.0):
		""" Choose an action and turn it into a move """
		chosen_action = self.pick_action(state, epsilon)
		#Get the transformation corresponding to the chosen action
		chosen_move = self.actions[chosen_action]
		#Return the name and transformation of the selected action
		return chosen_move, chosen_action		

	def q_learning(self, action, old_state, new_state, learning_rate, discount_factor, epsilon):
		#Get the q-value of the current state, action pair
		current_q_value = self.get_policy(old_state)[action]
		#Get the reward for the new state (10 if caught, 0 otherwise)
		reward = self.reward(new_state)
		#Get the max action for the new state
		softmax_backup = self.softmax
		self.softmax = False
		new_move, new_max_action = self.get_action(new_state, 0.0)
		self.softmax = softmax_backup
		#Get the q-value for the new state and its max action
		new_q_value = self.get_policy(new_state)[new_max_action]
		discounted_next = discount_factor*new_q_value
		difference_q = discounted_next - current_q_value
		#Update the q_value for the current state by adding reward + discounted next q-value - current q-value, discounted by learning rate
		updated_q_value = current_q_value + learning_rate * (reward + discount_factor * new_q_value - current_q_value)
		#Update the q-value for the old state, action pair
		self.get_policy(old_state)[action] = updated_q_value

	def get_new_location(self, object_location, transformation):
		""" Returns new location of an object when performs the chosen move """
		new_location = []
		#Retrieve the agent's position in the grid
		#Get the size of the environment
		environment_size = self.grid_size
		#Wrap edges to make grid toroidal
		new_location.append((object_location[0] + transformation[0]) % environment_size[0])
		new_location.append((object_location[1] + transformation[1]) % environment_size[1])
		return new_location

	def reward(self, state):
		if state[0] == state[2] and state[1] == state[3]:
			return 10
		else:
			return 0

	def pick_action(self, state, action_selection_var): #def pick_action(self, state, epsilon, e_greedy=True, softmax=False):
		""" Use the probabilities in the policy to pick a move """
		#Retrieve the policy for the current state using e_greedy or softmax
		# Note: action_selection_var is epsilon for e-greedy and temperature for softmax!
		if self.softmax == True and self.prey==False:
			policy = self.get_softmax_action_selection(self.get_policy(state), action_selection_var)
		else:
			policy = self.get_e_greedy_policy(self.get_policy(state), action_selection_var)

		#Zip the policy into a tuple of names, and a tuple of values
		action_name, policy = zip(*policy.items())
		#Use np.random.choice to select actions according to probabilities
		choice_index = np.random.choice(list(action_name), 1, p=list(policy))[0]
		#Return name of action
		return choice_index	

	def get_e_greedy_policy(self, policy, epsilon=0.0):
		"""
		With epsilon-greedy, at each time step, the agent selects a random
		action with a fixed probability, 0 <= epsilon<= 1, instead of selecting greedily one of
		the learned optimal actions with respect to the Q-function
		"""
		#Get |A(s)|
		number_actions = len(policy)
		#Get the extra probability to divide over actions
		extra_probability = epsilon/number_actions
		best_action_list = []
		other_action_list = []
		#Get the maximum value in the policy
		max_value = max(policy.iteritems(), key=operator.itemgetter(1))[1]

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
	
	def get_softmax_action_selection(self, policy, temperature=0.3):
		"""
		Softmax utilizes action-selection probabilities which are determined by
		ranking the value-function estimates using a Boltzmann distribution
		"""
		#For each action, divide by temperature
		softmax_prob = dict.fromkeys(policy.keys(), None)
		action_selection = []
		total_sum = 0
		sum_q_values = sum(policy.values())
		nr_of_q_values = len(policy)
		mean = sum_q_values/nr_of_q_values
		#print 'sum q values: ', sum_q_values, ' nr q values: ', nr_of_q_values,  ' mean: ', mean 
		for actionname, q_value in policy.iteritems():
			#print "q_value: ", q_value, " temperature: ", temperature
			#print 'q_value - mean = ', q_value - mean
			new_q  = math.exp((q_value-mean)/temperature) # used to be q_value
			#print 'new q: ', new_q
			action_selection.append((actionname,new_q))
			total_sum += new_q
		# Calculate softmax probabilities for each action
		for actionname, new_q in action_selection:
			value = new_q/total_sum
			softmax_prob[actionname] = value
		return softmax_prob
