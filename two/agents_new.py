import numpy as np
import copy 
import random

class Agent(object):
	"""Agent class, with policy"""
	def __init__(self, policy_grid=None, policy_given=False):
		if policy_grid is not None:
			self.policy_grid = policy_grid
		self.reward = 0
		self.policy_given = policy_given

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

	def get_action(self, state):
		return self.policy_grid.get_action(state)

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


class Predator(Agent):
	def __init__(self, policy):
		Agent.__init__(self, policy)

	def __repr__(self):
		""" Represent Predator as X """
		return ' X '		

class Prey(Agent):
	def __init__(self, policy):
		Agent.__init__(self, policy)

	def __repr__(self):
		""" Represent Prey as O """
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

