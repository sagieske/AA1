import numpy as np
import copy
import itertools
import math
import operator
import sys
import operator
import random
import helpers

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
		return [self.predator_location[0], self.predator_location[1], self.prey_location[0], self.prey_location[1]]

	def get_location(self, object_name):
		""" Retrieve location of agent in grid """
		#Check which agent, and return its location
		if(object_name == 'predator'):
			return self.predator_location
		elif(object_name == 'prey'):
			return self.prey_location

class Policy:
	""" Policy object that stores action values for each state """
	def __init__(self, grid_size, policy_grid=None, prey=False, softmax=False, amount_agents=2):
		""" Initialize policy object of certain grid_size, with optional initial policy_grid and True for a prey """
		#Store prey boolean
		self.prey = prey
		#Store on-policy MC boolean
		#Store grid size
		self.grid_size = grid_size
		#If the agent is not a prey, set the policy to random with initialization value
		self.policy = {'North':15, 'East':15, 'South':15, 'West':15, 'Wait':15}
		#If a policy_grid is given, store it
		if(policy_grid is not None):
			self.policy_grid = policy_grid
			#self.distance_dict = policy_grid
		else:
			index_numbers = []
			#for agent_number in range(0, amount_agents):
			#	index_numbers.append(agent_number)
			print amount_agents
			party_dict = np.empty((amount_agents,11,11), dtype=dict)
			party_dict.fill(copy.deepcopy(self.policy))
			print "dikke fissa"
			print party_dict
			#print index_numbers


			#self.policy_grid = [[[[copy.deepcopy(self.policy) for i in range(0, self.grid_size[1])] for j in range(0, self.grid_size[0])] for k in range(0, self.grid_size[1])] for l in range(0, self.grid_size[0])]	
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

			# Initialize q-values for all possible distances (distance_state) from prey as dictionaries 
			# with possible new distances from that distance_state
			self.distance_dict = {}
			for i,j in distances:
				possible_states_dict = {}
				new_possible_states = [[i,j], [i-1,j], [i+1, j], [i, j-1], [i, j+1]]
				for possible_state in new_possible_states:
					# impossible states
					if possible_state[0] < 0 or possible_state[0] == self.grid_size[0] or possible_state[1] < 0 or possible_state[1] == self.grid_size[0]:
						continue
					# Initialize as 15
					possible_states_dict[tuple(possible_state)] = 15
				# Initialize total distance dictionary
				self.distance_dict[(i,j)] = possible_states_dict

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
		i = state[0]
		j = state[1]
		k = state[2]
		l = state[3]
		return self.policy_grid[i][j][k][l]	

	def set_distance_dict(self, distance_dict):
		self.distance_dict = distance_dict

	def get_action(self, state, epsilon=0.0):
		""" Choose an action and turn it into a move """
		chosen_action = self.pick_action(state, epsilon)
		#Get the transformation corresponding to the chosen action
		chosen_move = self.actions[chosen_action]
		#Return the name and transformation of the selected action
		return chosen_move, chosen_action		

	def q_learning(self, action, old_state, new_state, learning_rate, discount_factor, epsilon):

		# Get current qvalue of movement from current distance state to new distance state
		current_xy = helpers.xy_distance(old_state, self.grid_size)
		new_xy = helpers.xy_distance(new_state,  self.grid_size)
		test_current_q_value = self.distance_dict[tuple(current_xy)][tuple(new_xy)]
		#Get the q-value of the current state, action pair
		#current_q_value = self.get_policy(old_state)[action]

		#Get the reward for the new state (10 if caught, 0 otherwise)
		reward = self.reward(new_state)
		#Get the max action for the new state
		softmax_backup = self.softmax
		self.softmax = False
		new_move, new_max_action = self.get_action(new_state, 0.0)
		self.softmax = softmax_backup
		#Get the q-value for the new state and its max action
		#new_q_value = self.get_policy(new_state)[new_max_action]


		# Get new location using the maximal action in the new state
		max_location_new_state = self.get_new_location([new_state[0], new_state[1]], new_move)
		# Get its distance state
		max_xy_new_state = helpers.xy_distance([max_location_new_state[0], max_location_new_state[1], new_state[2], new_state[3]],  self.grid_size)

		# Get the q_value of this max distance state from the new distance state
		test_new_q_value = self.distance_dict[tuple(new_xy)][tuple(max_xy_new_state)]
		#new_xy = helpers.xy_distance(new_state,  self.grid_size)


		#discounted_next = discount_factor*new_q_value
		#difference_q = discounted_next - current_q_value
		#Update the q_value for the current state by adding reward + discounted next q-value - current q-value, discounted by learning rate
		#updated_q_value = current_q_value + learning_rate * (reward + discount_factor * new_q_value - current_q_value)
		#Update the q-value for the old state, action pair
		#self.get_policy(old_state)[action] = updated_q_value


		test_updated_q_value = test_current_q_value + learning_rate * (reward + discount_factor * test_new_q_value - test_current_q_value)

		# Update in distance dictionary
		self.distance_dict[tuple(current_xy)][tuple(new_xy)] = test_updated_q_value


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
		current_xy = helpers.xy_distance(state, self.grid_size)

		dist_to_action = helpers.distance_to_action(state)

		test_policy = {}
		#for key, value in self.distance_dict[tuple(current_xy)].iteritems():
		for key,value in dist_to_action.iteritems():
			test_policy[key] = self.distance_dict[tuple(current_xy)][tuple(value[0])]

		if self.softmax == True and self.prey==False:
			#policy = self.get_softmax_action_selection(self.get_policy(state), action_selection_var)
			policy = self.get_softmax_action_selection(test_policy, action_selection_var)
		else:
			#policy = self.get_e_greedy_policy(self.get_policy(state), action_selection_var)
			policy = self.get_e_greedy_policy(test_policy, action_selection_var)

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
