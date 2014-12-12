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
import pulp

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
		print 'Policy: learning_type:', learning_type
		""" Initialize policy object of certain grid_size, with optional initial policy_grid and True for a prey """
		print "number of agents: ", amount_agents
		self.agent_name = agent_name
		#Store prey boolean
		self.prey = prey
		#Store on-policy MC boolean
		#Store grid size
		self.grid_size = grid_size
		self.learning_type = learning_type
		if learning_type == 'Minimax':
			init_value = 1.0
		else:
			init_value = 15.0
	
	
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
					if learning_type == 'Minimax':
				                new_possible_states = [ [i-2,j],
             			                                        [i-1, j-1], [i-1,j], [i-1, j+1],
             			                                        [i, j-2], [i, j-1], [i,j], [i, j+1], [i, j+2],
             			                                        [i+1, j-1], [i+1, j], [i+1, j+1],
             			                                        [i+2, j]]
             			        else:
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
				# Initialize every item to initial value
				for item in possible_new_state_combos:
					inner_dict[item] = init_value
				# Add this dictionary of new possible distances as value to party dict where key is current distance to other agents.
				self.party_dict[key] = inner_dict



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


	def return_state_policy(self,s):
		new_state, agent_name = self.state_dict_to_state_distances(s)
		policy = self.get_encoded_policy(new_state)
		policy, policy_with_action = helpers.get_feasible_actions(copy.deepcopy(s), agent_name, policy, grid_size=self.grid_size)
		return policy

	def q_learning(self, a, s, s_prime, learning_rate, discount_factor, epsilon, agent_list, reward_list):

		new_state, agent_name = self.state_dict_to_state_distances(s)

		#Get policy encoded
		policy = self.get_encoded_policy(new_state)
		#Get reachable states fr0om this state
		policy, policy_with_action = helpers.get_feasible_actions(copy.deepcopy(s), agent_name, policy, grid_size=self.grid_size)

		#Get current Q-value
		current_q = policy[new_state]
		#Get reward
		reward = reward_list[int(agent_name)]
		softmax_backup = self.softmax
		self.softmax = False
		#Get the max action for the new state
		new_max_action, new_move = self.get_action(s_prime, 0.0)
		
		self.softmax = softmax_backup

		#Get the q-value for the new state and its max action
		prime_agent_location = s_prime[agent_name]

		max_prime_agent_location = helpers.get_new_location(prime_agent_location, new_move, grid_size=self.grid_size)


		new_state_dict = copy.deepcopy(s_prime)
		
		s_prime_distance = self.state_dict_to_state_distances(s_prime)[0]
	
		max_policy = self.get_encoded_policy(s_prime_distance)
		max_policy, max_policy_with_action = helpers.get_feasible_actions(copy.deepcopy(s_prime), agent_name, max_policy, grid_size=self.grid_size)
		max_q = max_policy[s_prime_distance]


		
		# Transform current location to new location using chosen action a
		agent_new_state = helpers.get_new_location(s[agent_name], self.actions[a])
		# Calculate the distance to all other agents from this chosen distance (while they stand still)
		new_distances_to_agents = helpers.get_all_distances_to_agents(agent_name, agent_new_state, s)

		# Update in distance dictionary
		updated_q_value = current_q + learning_rate * (reward + discount_factor * max_q - current_q)
		self.party_dict[new_state][new_distances_to_agents] = updated_q_value

		return self.party_dict[new_state][new_distances_to_agents]		


	def sarsa(self, a, s, s_prime, learning_rate, discount_factor, epsilon, agent_list, reward_list):
                
		new_state, agent_name = self.state_dict_to_state_distances(s)

		#Get policy encoded
		policy = self.get_encoded_policy(new_state)
		#Get reachable states fr0om this state

		policy, policy_with_action = helpers.get_feasible_actions(copy.deepcopy(s), agent_name, policy,grid_size=self.grid_size)

		#Get current Q-value
		current_q = policy[new_state]
		#Get reward
		reward = reward_list[int(agent_name)]
		softmax_backup = self.softmax
		self.softmax = False
		#Get the max action for the new state

		new_max_action, new_move = self.get_action(s_prime, epsilon)

		self.softmax = softmax_backup

		#Get the q-value for the new state and its max action
		prime_agent_location = s_prime[agent_name]
		max_prime_agent_location = self.get_new_location(prime_agent_location, new_move)

		new_state_dict = copy.deepcopy(s_prime)
		
		s_prime_distance = self.state_dict_to_state_distances(s_prime)[0]

		max_policy = self.get_encoded_policy(s_prime_distance)
		
		max_policy, max_policy_with_action = helpers.get_feasible_actions(copy.deepcopy(s_prime), agent_name, max_policy, grid_size=self.grid_size)
		max_q = max_policy[s_prime_distance]

		updated_q_value = current_q + learning_rate * (reward + discount_factor * max_q - current_q)
		
		# Transform current location to new location using chosen action a
		agent_new_state = helpers.get_new_location(s[agent_name], self.actions[a])
		# Calculate the distance to all other agents from this chosen distance (while they stand still)
		new_distances_to_agents = helpers.get_all_distances_to_agents(agent_name, agent_new_state, s)

		# Update in distance dictionary
		self.party_dict[new_state][new_distances_to_agents] = updated_q_value
		return new_move, new_max_action


		
	def Minimax_q_learning(self, a, opponent_action, s, s_prime, learning_rate, discount_factor, epsilon, reward_list):
		"""
		s:		dictionary of current state locations. Keys are agent IDs (str) and value is [x,y] location
		s': 		dictinary of new state locations. Keys are agent IDs (str) and value is new [x,y] location
		a: 		action name (str) of action taken by agent
		opponent_action: action name (str) of action taken by other agent
		"""
		# Get currentdistance states 
		cur_dist = self.state_dict_to_state_distances(s)[0]
		new_dist = self.state_dict_to_state_distances(s_prime)[0]

		current_state_dist, agent_name = self.state_dict_to_state_distances(s)

		#Get policy encoded using current distance state
		policy = self.get_encoded_policy(current_state_dist)

		#Get reachable states from this state
		policy, policy_with_action = helpers.get_feasible_actions(copy.deepcopy(s), agent_name, policy, grid_size=self.grid_size)

		#Get current Q-value
		current_q = policy[current_state_dist]

		#Get reward
		reward = reward_list[int(agent_name)]

		# Calculate value of s prie
		value = self.calculate_V(s_prime, agent_name)
		# Update q value in dictionary using (1-alpha) Q(s,a,o) + alpha * (R + discount V(s'))
		updated_q_value = current_q + learning_rate * (reward + discount_factor * value - current_q)
		self.party_dict[cur_dist][new_dist] = updated_q_value

		return self.party_dict[cur_dist][new_dist]





	def calculate_V(self, s, agent_name):
		""" Calculate V for state s by maximizing it while minimizing opponents actions. Returns the maximium value of V
		"""
		max_v = pulp.LpProblem("Maximize V",  pulp.LpMaximize)

		# Set V as variable to maximize
		v  = pulp.LpVariable("v", 0.0, cat="Continuous")
		max_v += v
		actions = ['West', 'East','North', 'South','Wait']
		# Create policy var for actions
		action_policy_vars = pulp.LpVariable.dicts("A",actions,lowBound =0.0, upBound = 1.0, cat="Continuous")

		# Probabilities sum to 1
		max_v += sum([action_policy_vars[a] for a in actions]) == 1
		for a in actions:
			max_v += action_policy_vars[a] >= 0.000000001 

		# add constraints as summation of actions given an opponent action are bigger than 0
		for o in actions:
			max_v += sum([self.get_qvalue_minimax(s, agent_name, a, o) * action_policy_vars[a] for a in actions]) >= v

		# Solve maximization
		max_v.solve()
		#for i in actions:
		#	if action_policy_vars[i].value() == 1.0:
		#		print i


		return pulp.value(max_v.objective)

	def get_qvalue_minimax(self, state, agent_name, action, opponent_action):
		"""
		Returns q value for specific state using action and opponent_action
		"""
		new_state = copy.deepcopy(state)
		# Get new location for agent
		agent_new_state = helpers.get_new_location(state[agent_name], self.actions[action], grid_size=self.grid_size)
		new_state[agent_name] = agent_new_state
		# Get name of other agent
		other_agents = state.keys()
		del other_agents[other_agents.index(agent_name)]
		other_agents_name = other_agents[0]

		# Get new location for other agent
		other_agent_new_state = helpers.get_new_location(state[other_agents_name], self.actions[opponent_action], grid_size=self.grid_size)
		new_state[other_agents_name] = other_agent_new_state

		# Get current and new distance states 
		cur_dist = self.state_dict_to_state_distances(state)[0]
		new_dist = self.state_dict_to_state_distances(new_state)[0]
		# Get q value
		q_value = self.party_dict[cur_dist][new_dist]

		return q_value

		

	def get_new_location(self, object_location, transformation, grid_size=[11,11]):
		""" Returns new location of an object when performs the chosen move """
		new_location = []
		#Retrieve the agent's position in the grid
		#Get the size of the environment
		environment_size = grid_size
		#Wrap edges to make grid toroidal
		new_location.append((object_location[0] + transformation[0]) % environment_size[0])
		new_location.append((object_location[1] + transformation[1]) % environment_size[1])
		return new_location

	def reward(self, state):
		if state[0] == state[2] and state[1] == state[3]:
			return 10
		else:
			return 0

	def state_dict_to_state_distances(self, state_dict):
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

	def get_action(self, state, action_selection_var, minimax=False): #t
		""" Use the probabilities in the policy to pick a move """
		#Retrieve the policy for the current state using e_greedy or softmax
		# Note: action_selection_var is epsilon for e-greedy and temperature for softmax!
		new_state, agent_name = self.state_dict_to_state_distances(state)
		#dist_to_action = helpers.distance_to_action(new_state, self.agent_name, self.location_dict)
		policy = self.get_encoded_policy(new_state)

		policy, policy_with_action = helpers.get_feasible_actions(copy.deepcopy(state), agent_name, policy, grid_size=self.grid_size)

		if self.softmax == True and self.prey==False:
			policy = self.get_softmax_action_selection(policy, action_selection_var)
		else:
			policy = self.get_e_greedy_policy(policy, action_selection_var)
		#Zip the policy into a tuple of names, and a tuple of values
		action_name, policy = zip(*policy.items())

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
		for actionname, q_value in policy.iteritems():
			new_q  = math.exp((q_value-mean)/temperature) # used to be q_value
			action_selection.append((actionname,new_q))
			total_sum += new_q
		# Calculate softmax probabilities for each action
		for actionname, new_q in action_selection:
			value = new_q/total_sum
			softmax_prob[actionname] = value
		return softmax_prob
