import numpy as np
import copy
import itertools
import math
import operator
import sys
import operator
import random
import helpers
import ast
import pdb

class Environment:
	""" Grid object that stores agent locations and state """
	def __init__(self, grid_size, location_dict):
		""" Initialize environment of given size and predator, prey locations """
		self.grid_size = grid_size
		#Create empty grid
		self.grid = [[ ' ' for i in range(0, grid_size[0])] for y in range(0, grid_size[1])]
		#Store predator, prey locations
		self.location_dict = location_dict

		for agent_name in location_dict.keys():
			location = location_dict[agent_name]
			# Agent 0 is prey, all other agents are the predators
			if agent_name == "0":
				self.grid[location[0]][location[1]] = 'O'
			else:
				self.grid[location[0]][location[1]] = 'X'

	def print_grid(self):
		""" Print the environment """
		print "=========="
		for row in self.grid:
			print row
		print "=========="

	def place_object(self, agent_name, new_location):
		""" Place an object at a given location in the environment"""
		if(agent_name=='0'):
			#Update grid
			self.grid[new_location[0]][new_location[1]] = 'O'
		else:
			self.grid[new_location[0]][new_location[1]] = 'X'
		#Update location dict
		self.location_dict[agent_name] = new_location

	def move_object(self, agent_name, new_location):
		""" Move object from old to new location in the grid """
		old_location = self.location_dict[agent_name]
		# First overwrite old location
		self.grid[old_location[0]][old_location[1]] = ' '

		# Then write to new location		
		if(agent_name == '0'):
			self.grid[new_location[0]][new_location[1]] = '0'
		else:
			self.grid[new_location[0]][new_location[1]] = 'X'
		
		#Update location dict
		self.location_dict[agent_name] = new_location		

	def get_size(self):
		""" Return environment size"""
		return self.grid_size

	def get_state(self):
		""" Return the current state that the environment's in """
		return self.location_dict

	def get_location(self, object_name):
		""" Retrieve location of agent in grid """
		#Check which agent, and return its location
		return self.location_dict[object_name]

class Policy:
	""" Policy object that stores action values for each state """
	def __init__(self, grid_size, policy_grid=None, prey=False, softmax=False, amount_agents=2, agent_name=None, learning_type='Q-learning'):
		""" Initialize policy object of certain grid_size, with optional initial policy_grid and True for a prey """
		print "number of agents: ", amount_agents
		self.agent_name = agent_name
		#Store prey boolean
		self.prey = prey
		#Store on-policy MC boolean
		#Store grid size
		self.grid_size = grid_size
		self.learning_type = learning_type
		
		
		
		#If the agent is not a prey, set the policy to random with initialization value
		
		if learning_type == 'Minimax':
			self.policy = { 'North': {'North':15, 'East':15, 'South':15, 'West':15, 'Wait':15},
             			        'East': {'North':15, 'East':15, 'South':15, 'West':15, 'Wait':15},
             			        'South': {'North':15, 'East':15, 'South':15, 'West':15, 'Wait':15},
             			        'West': {'North':15, 'East':15, 'South':15, 'West':15, 'Wait':15},
             			        'Wait': {'North':15, 'East':15, 'South':15, 'West':15, 'Wait':15}}
			#self.distance_dict = policy_grid
		else:
			self.policy = {'North':15, 'East':15, 'South':15, 'West':15, 'Wait':15}
		
		
		#If a policy_grid is given, store it
		if(policy_grid is not None):
			self.policy_grid = policy_grid
			#self.distance_dict = policy_grid
		else:
			index_numbers = []
			#for agent_number in range(0, amount_agents):
			#	index_numbers.append(agent_number)
			party_dict = np.empty((amount_agents,11,11), dtype=dict)
			party_dict.fill(copy.deepcopy(self.policy))
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
					possible_states_dict[tuple(possible_state)] = 15.0
				# Initialize total distance dictionary
				self.distance_dict[(i,j)] = possible_states_dict

	
				
			#Create all possible combination for distances
			distances_agent = list(itertools.product(range_max_distance_x , range_max_distance_y))
			list_distance_agent = [distances_agent] * (amount_agents-1)
			# Create all possible combinations for distances to other agents
			total_combos = list(itertools.product(*list_distance_agent))

			self.party_dict = {}
			# loop over all possible combination of distances to other agents and use them as keys in party_dict 
			for distance_to_rest in total_combos:
				key = distance_to_rest
				# initialize inner dict for that key
				inner_dict= {}
				tuple_array = list(distance_to_rest)
				# Array to store arrays (# agents-10) and in those arrays the possible new distances to that agent
				possible_variations = []
				# Get all possible new state combinations from current state combo (key)
				for index in range(0, len(tuple_array)):
					i,j = tuple_array[index]
					distance_changes = []
					# get possible states
					new_possible_states = [[i,j], [i-1,j], [i+1, j], [i, j-1], [i, j+1]]
					for possible_state in new_possible_states:
						# impossible distance
						if possible_state[0] < 0 or possible_state[0] == self.grid_size[0] or possible_state[1] < 0 or possible_state[1] ==self.grid_size[0]:
							continue		
						# Add as possible new state
						distance_changes.append(tuple(possible_state))		
					
					# Add possible variation in this state to array
					possible_variations.append(distance_changes)

				# Get all possible combinations from array of possible new distances to other agents
				possible_new_state_combos = list(itertools.product(*possible_variations))
				# Initialize every item to 15
				for item in possible_new_state_combos:
					inner_dict[item] = 15.0
				# Add this dictionary of new possible distances as value to party dict where key is current distance to other agents.
				self.party_dict[key] = inner_dict

			#for key, value in self.party_dict.iteritems():
			#	print "key:", key
			#	print "possible new states: ", value
			#	print "-"
			#	print len(value)
			

		self.softmax = softmax




	def update_Q_values(self,Q, params):
		if self.learning_type == 'Minimax':
			state = params[0]
			i = state[0]
			j = state[1]
			k = state[2]
			l = state[3]
			action = params[1]
			opponent_action = params[2] 
			self.policy_grid[i][j][k][l][action][opponent_action] = Q
		else:
			state = params[0]
			state = params[0]
			i = state[0]
			j = state[1]
			k = state[2]
			l = state[3]
			action = params[1]
			self.policy_grid[i][j][k][l][action] = Q



        # Stays the same for Minimax
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
		#chosen_action = self.pick_action(state, epsilon)
		#Get the transformation corresponding to the chosen action
		#chosen_name, chosen_move = helpers.distance_to_action(state, self.agent_name, ast.literal_eval(chosen_action))


		chosen_name, chosen_move = self.pick_action(state, epsilon)	

		#Return the name and transformation of the selected action
		return chosen_move, chosen_name		

	def return_state_policy(self,s):
		new_state, agent_name = self.dict_to_state(s)
		policy = self.get_encoded_policy(new_state)
		policy, policy_with_action = helpers.get_feasible_actions(copy.deepcopy(s), agent_name, policy)
		return policy

	def q_learning(self, a, s, s_prime, learning_rate, discount_factor, epsilon, agent_list, reward_list):

		new_state, agent_name = self.dict_to_state(s)

		#Get policy encoded
		policy = self.get_encoded_policy(new_state)
		#Get reachable states from this state
		policy, policy_with_action = helpers.get_feasible_actions(copy.deepcopy(s), agent_name, policy)

		#Get current Q-value
		current_q = policy[new_state]
		#Get reward
		reward = reward_list[int(agent_name)]
		softmax_backup = self.softmax
		self.softmax = False
		#Get the max action for the new state
		new_move, new_max_action = self.get_action(s_prime, 0.0)
		
		self.softmax = softmax_backup

		#Get the q-value for the new state and its max action
		#new_q_value = self.get_policy(new_state)[new_max_action]
		prime_agent_location = s_prime[agent_name]
		max_prime_agent_location = self.get_new_location(prime_agent_location, new_move)
		#print "old: ", prime_agent_location
		#print "max: ", max_prime_agent_location

		new_state_dict = copy.deepcopy(s_prime)
		#print "oldstdict ", s, " distance: ", new_state
		
		s_prime_distance = self.dict_to_state(s_prime)[0]
		#print "newstdict ", s_prime, " distance ", s_prime_distance

		#NOT NEEDED: s_prime[agent_name] = max_prime_agent_location
		#print  "current: ", new_state

		max_policy = self.get_encoded_policy(s_prime_distance)
		#print "max pol: ", max_policy


		
		max_policy, max_policy_with_action = helpers.get_feasible_actions(copy.deepcopy(s_prime), agent_name, max_policy)
		max_q = max_policy[s_prime_distance]

		updated_q_value = current_q + learning_rate * (reward + discount_factor * max_q - current_q)

		# Transform current location to new location using chosen action a
		agent_new_state = helpers.get_new_location(s[agent_name], self.actions[a])
		# Calculate the distance to all other agents from this chosen distance (while they stand still)
		new_distances_to_agents = helpers.get_all_distances_to_agents(agent_name, agent_new_state, s)

		# Update in distance dictionary
		#print "old Q value:", self.party_dict[new_state][new_distances_to_agents]
		#print "updated Q: ", updated_q_value
		#print "new state: ", new_state
		#print "---------"
		#print "party_dict[%s][%s] " %(str(new_state), str(new_distances_to_agents) )
		self.party_dict[new_state][new_distances_to_agents] = updated_q_value

		#print "found: ", self.party_dict[new_state][new_distances_to_agents]
		return self.party_dict[new_state][new_distances_to_agents]
		
		
	def Minimax_q_learning(self, a, s, s_prime, learning_rate, discount_factor, epsilon, agent_list, reward_list):

		new_state, agent_name = self.dict_to_state(s)

		#Get policy encoded
		policy = self.get_encoded_policy(new_state)
		#Get reachable states from this state
		policy, policy_with_action = helpers.get_feasible_actions(copy.deepcopy(s), agent_name, policy)

		#Get current Q-value
		current_q = policy[new_state]
		#Get reward
		reward = reward_list[int(agent_name)]
		softmax_backup = self.softmax
		self.softmax = False
		#Get the max action for the new state
		new_move, new_max_action = self.get_action(s_prime, 0.0)
		
		self.softmax = softmax_backup

		#Get the q-value for the new state and its max action
		#new_q_value = self.get_policy(new_state)[new_max_action]
		prime_agent_location = s_prime[agent_name]
		max_prime_agent_location = self.get_new_location(prime_agent_location, new_move)
		#print "old: ", prime_agent_location
		#print "max: ", max_prime_agent_location

		new_state_dict = copy.deepcopy(s_prime)
		#print "oldstdict ", s, " distance: ", new_state
		
		s_prime_distance = self.dict_to_state(s_prime)[0]
		#print "newstdict ", s_prime, " distance ", s_prime_distance

		#NOT NEEDED: s_prime[agent_name] = max_prime_agent_location
		#print  "current: ", new_state

		max_policy = self.get_encoded_policy(s_prime_distance)
		#print "max pol: ", max_policy


		
		max_policy, max_policy_with_action = helpers.get_feasible_actions(copy.deepcopy(s_prime), agent_name, max_policy)
		max_q = max_policy[s_prime_distance]

		updated_q_value = current_q + learning_rate * (reward + discount_factor * max_q - current_q)

		# Transform current location to new location using chosen action a
		agent_new_state = helpers.get_new_location(s[agent_name], self.actions[a])
		# Calculate the distance to all other agents from this chosen distance (while they stand still)
		new_distances_to_agents = helpers.get_all_distances_to_agents(agent_name, agent_new_state, s)

		# Update in distance dictionary
		#print "old Q value:", self.party_dict[new_state][new_distances_to_agents]
		#print "updated Q: ", updated_q_value
		#print "new state: ", new_state
		#print "---------"
		#print "party_dict[%s][%s] " %(str(new_state), str(new_distances_to_agents) )
		self.party_dict[new_state][new_distances_to_agents] = updated_q_value

		#print "found: ", self.party_dict[new_state][new_distances_to_agents]
		return self.party_dict[new_state][new_distances_to_agents]


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

	def dict_to_state(self, state_dict):
		""" Calculate distance to all other agents using the dictionary of locations of all agents(=state)
		Returns tuple of distances to other agents from current agent and ID of current agent
		"""
		# Get all names from agent and sort to always loop through agents in ascending ID order
		agent_list = state_dict.keys()
		agent_list.sort()
		# Initialize tuple for states
		state_tuple = ()

		# For every other agent get distance from current agent and append to state tuple		
		for other_agent in agent_list:
			if(other_agent != self.agent_name):
				# Create state array 4x1
				state_list = [state_dict[other_agent][0], state_dict[other_agent][1], state_dict[self.agent_name][0], state_dict[self.agent_name][1]]
				# Calculate distance to agent
				distance_to_other = helpers.xy_distance(state_list, self.grid_size)
				# Add to tuple for state
				state_tuple += (tuple(distance_to_other),)
		return state_tuple, self.agent_name

	def tuple_to_old_state(self, tup):
		return [tup[0], tup[1]]

	def get_encoded_policy(self, state):
		return self.party_dict[state]

	def pick_action(self, state, action_selection_var): #t
		""" Use the probabilities in the policy to pick a move """
		#Retrieve the policy for the current state using e_greedy or softmax
		# Note: action_selection_var is epsilon for e-greedy and temperature for softmax!
		new_state, agent_name = self.dict_to_state(state)
		#dist_to_action = helpers.distance_to_action(new_state, self.agent_name, self.location_dict)
		policy = self.get_encoded_policy(new_state)

		policy, policy_with_action = helpers.get_feasible_actions(copy.deepcopy(state), agent_name, policy)

		#test_policy = {}
		#for key, value in self.distance_dict[tuple(current_xy)].iteritems():
		#for key,value in dist_to_action.iteritems():
		#	test_policy[key] = self.distance_dict[tuple(current_xy)][tuple(value[0])]
		if self.softmax == True and self.prey==False:
			#policy = self.get_softmax_action_selection(self.get_policy(state), action_selection_var)
			policy = self.get_softmax_action_selection(policy, action_selection_var)
		else:
			#policy = self.get_e_greedy_policy(self.get_policy(state), action_selection_var)
			policy = self.get_e_greedy_policy(policy, action_selection_var)

		#Zip the policy into a tuple of names, and a tuple of values
		action_name, policy = zip(*policy.items())
#		print "action_name: ", list(action_name)
		#Use np.random.choice to select actions according to probabilities
		choice_index = np.random.choice(list(action_name), 1, p=list(policy))[0]
		choice_index = ast.literal_eval(choice_index)
		#Get the action name
		action_name = policy_with_action.get(choice_index, None)

		#Return name of action and corresponding transformation
		return action_name, self.actions[action_name]	

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
				best_action_list.append(str(action[0]))
			#Otherwise, append to other_action_list
			else:
				other_action_list.append(str(action[0]))
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
