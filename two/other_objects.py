import numpy as np
import copy
import itertools
import math
import operator
import sys
import operator

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
	def __init__(self, grid_size, policy_grid=None, prey=False, verbose=0):
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
			self.policy_grid = [[[[self.policy for i in range(0, self.grid_size[1])] for j in range(0, self.grid_size[0])] for k in range(0, self.grid_size[1])] for l in range(0, self.grid_size[0])]
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
		self.verbose= verbose

				
	def get_policy(self, state):
		""" Return the policy dictionary for a state """
		i = state[0]
		j = state[1]
		k = state[2]
		l = state[3]
		return self.policy_grid[i][j][k][l]

	def get_action(self, state, epsilon=0.0, restricted=None):
		""" Choose an action and turn it into a move """
		if self.verbose > 0:
			print "epsilon: ", epsilon
			print "state: ", state
		#If there are restricted actions, do a restricted action pick
		if restricted is not None:
			chosen_action = self.pick_action_restricted(state, epsilon, restricted)
		#Otherwise, just choose an action
		else:
			chosen_action = self.pick_action(state, epsilon)
		#Get the transformation corresponding to the chosen action
		chosen_move = self.actions[chosen_action]
		#Return the name and transformation of the selected action
		return chosen_move, chosen_action		

	def q_learning(self, action, old_state, new_state, learning_rate, discount_factor, epsilon):
		current_q_value = self.get_policy(old_state)[action]
		reward = self.reward(old_state)
		new_move, new_max_action = self.get_action(new_state, 0.0)
		new_predator_location = self.get_new_location([new_state[0], new_state[1]], new_move)
		new_max_state = [new_predator_location[0], new_predator_location[1], old_state[2], old_state[3]]
		new_q_value = self.get_policy(new_state)[new_max_action]
		updated_q_value = current_q_value + learning_rate * (reward + discount_factor * new_q_value - current_q_value)
		self.get_policy(old_state)[action] = updated_q_value
		if self.verbose > 0:
			print "In state ", old_state, " action ", action, " was chosen leading to state ", new_state
			print "Q value for this state is ", current_q_value, " epsilon is ", epsilon, " reward is ", reward
			print "New max action = ", new_max_action, " with new move: ", new_move
			print "New predator_location: ", new_predator_location
			print "New max state: ", new_max_state
			print "Q value for next state is ", new_q_value
			print "Update q value for state ", old_state, " and action ", action, " is ", updated_q_value

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

	def pick_action(self, state, epsilon, e_greedy=False, softmax=True):
		""" Use the probabilities in the policy to pick a move """
		#Retrieve the policy for the current state using e_greedy or softmax
		if e_greedy:
			policy = self.get_e_greedy_policy(self.get_policy(state), epsilon)
		# softmax
		elif softmax:
			policy =self.get_softmax_action_selection(self.get_policy(state))
		else:
			print "YOU DID NOT CHOOSE AN ACTION-SELECTION METHOD, STUPID YOU.."


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
		for actionname, q_value in policy.iteritems():
			new_q  = math.exp(q_value/temperature)
			action_selection.append((actionname,new_q))
			total_sum += new_q
		# Calculate softmax probabilities for each action
		for actionname, new_q in action_selection:
			value = new_q/total_sum
			softmax_prob[actionname] = value
		return softmax_prob


	def pick_action_restricted(self, state, epsilon, blocked_moves):
		""" Use the probabilities in the policy to pick a move but can not perform blocked move """
		#Make a deep copy of the policy to prevent accidental pops
		temp_policy = copy.deepcopy(self.get_e_greedy_policy(self.get_policy(state), epsilon))
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
