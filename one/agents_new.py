import numpy as np
import copy 
import random

class Predator:
	"""Predator class, with policy"""
	def __init__(self, location, prey_predator_distance, gridsize=[11,11], policy=None, policy_given=False):
		self.policy_dict = {'North':0.2, 'East':0.2, 'South':0.2, 'West':0.2, 'Wait':0.2}
		if(policy is not None):
			self.policy = policy
		else:
			self.policy = [[[[self.policy_dict for i in range(0, gridsize[1])] for j in range(0, gridsize[0])] for k in range(0, gridsize[1])] for l in range(0, gridsize[0])]
		self.actions = {'North': [-1,0], 'East': [0,1], 'South': [1,0],'West': [0,-1], 'Wait':[0,0]}
		self.location = location
		self.state = "Predator(" + str(self.location[0]) + "," + str(self.location[1]) + ")"
		self.reward = 0
		self.prey_predator_distance = prey_predator_distance
		self.policy_given = policy_given

	def __repr__(self):
		""" Represent Predator as X """
		return ' X '

	def action(self, state):
		""" Choose an action and turn it into a move """
		chosen_action = self.pick_action(state)
		chosen_move = self.actions[chosen_action]
		return chosen_move, chosen_action

	def pick_action(self, state):
		""" Use the probabilities in the policy to pick a move """
		policy = self.get_policy(state)
		action_name, policy = zip(*policy.items())
		# Get choice using probability distribution
		choice_index = np.random.choice(list(action_name), 1, p=list(policy))[0]
		return choice_index

	def get_location(self):
		""" Returns location of predator """
		return self.location

	def set_location(self, new_location):
		""" Set location of predator """
		self.location = new_location
		self.set_state(new_location)

	def get_state(self):
		""" Get state of predator """
		return self.state

	def set_state(self, new_location):
		""" Set state of predator """
		self.state = "Predator(" + str(new_location[0]) + "," + str(new_location[1]) + ")"	

	def get_transformation(self, action):
		""" Get transformation vector ([0 1], [-1 0], etc) given an action ('North', 'West', etc.) """
		return self.actions[action]
                
	def update_reward(self, reward):
		""" Add reward gained on time step to total reward """
		self.reward += reward

	def reset_reward(self):
		""" Reset reward to inital value """
		self.reward = 0

	def get_reward(self):
		""" Get collected reward for predator """
		return self.reward

	def get_policy(self, state):
		""" Return the predator's policy """
		i = state[0]
		j = state[1]
		k = state[2]
		l = state[3]
		return self.policy[i][j][k][l]

	def get_policy_grid(self):
		return self.policy

	def set_policy_grid(self, policy_grid):
		self.policy = policy_grid
		
	def get_action_keys(self):
		return self.get_policy().keys() 		



class Prey:
	"""Prey class, with policy"""
	def __init__(self, location, policy=None):
		""" Initialize Prey with standard policy """
		if(policy is not None):
			self.policy = policy
		else:
			self.policy = {'North':0.05, 'East':0.05, 'South':0.05, 'West':0.05, 'Wait':0.8}
		self.actions = {'North': [-1,0], 'East': [0,1], 'South': [1,0],'West': [0,-1], 'Wait':[0,0]}
		self.location = location
		self.state = "Prey(" + str(self.location[0]) + "," + str(self.location[1]) + ")"

	def __repr__(self):
		""" Represent Prey as 0 """
		return ' O '

	def action(self, restricted=None):
		""" Choose an action and turn it into a move """
		# Check if restricted subset of moves can be chosen
		if restricted is not None:
			chosen_action = self.pick_action_restricted(restricted)
		else:
			chosen_action = self.pick_action()
		chosen_move = self.actions[chosen_action]
		return chosen_move, chosen_action

	def pick_action(self):
		""" Use the probabilities in the policy to pick a move """

		# Split policy dictionary in list of keys and list of values
		old_policy = copy.deepcopy(self.policy)
		action_name, policy = zip(*old_policy.items())
		# Get choice using probability distribution
		choice_index = np.random.choice(list(action_name), 1, p=list(policy))[0]
		return choice_index

	def pick_action_restricted(self, blocked_moves):
		""" Use the probabilities in the policy to pick a move but can not perform blocked move """
		# Temporary policy list
		temp_policy = copy.deepcopy(self.policy)
		# Keep track of probability of deleted moves
		update_probability = 0
		# Delete blocked moves from temporary policy list
		for block in blocked_moves:
			update_probability += temp_policy[block]
			del temp_policy[block]			

		# Split policy dictionary in list of keys and list of values
		action_name, policy = zip(*temp_policy.items())
		# Create new policy wrt deleted moves
		added_probability = update_probability/float(len(blocked_moves))
		new_policy = new_list = [x+added_probability for x in list(policy)]
		# Get choice using probability distribution
		choice_index = np.random.choice(list(action_name), 1, new_policy)[0]
		return choice_index

	def get_location(self):
		""" Return location of prey """
		return self.location		

	def set_location(self, new_location):
		""" Set location of prey """
		self.location = new_location
		self.set_state(new_location)
	
	def get_state(self):
		""" Return state of prey """
		return self.state	

	def set_state(self, new_location):
		""" Set state of prey """
		self.state = "Prey(" + str(new_location[0]) + "," + str(new_location[1]) + ")"	

	def get_policy(self):
		return self.policy

