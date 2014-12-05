import numpy as np
import copy 
import random

class Agent(object):
	""" Agent class, with policy """
	def __init__(self, policy_grid, index):
		#Store given policy
		self.policy_grid = policy_grid
		self.name = index
		#Set reward to 0

	def get_name(self):
		return self.name
		
	def get_transformation(self, action):
		""" Get transformation vector ([0 1], [-1 0], etc) given an action ('North', 'West', etc.) """
		return self.actions[action]

	def get_action(self, state, epsilon=0.0):
		#Retrieve an action using the policy for this state in the policy object 
		return self.policy_grid.get_action(state, epsilon)

	def get_policy(self, state):
		""" Return the predator's policy """
		#Get indices to retrieve policy
		i = state[0]
		j = state[1]
		k = state[2]
		l = state[3]
		return self.policy[i][j][k][l]

	def get_policy_grid(self):
		""" Return policy grid for agent """
		return self.policy_grid

	def set_policy_grid(self, policy_grid):
		""" Set policy grid for agent """
		self.policy = policy_grid
		
	def get_action_keys(self, state):
		""" Return the names of the actions for a state """
		return self.get_policy(state).keys() 		

	def q_learning(self, action, old_state, new_state, learning_rate, discount_factor, epsilon):
		self.policy_grid.q_learning(action, old_state, new_state, learning_rate, discount_factor, epsilon)

	def sarsa(self, action, old_state, new_state, learning_rate, discount_factor, epsilon):
		self.policy_grid.sarsa(action, old_state, new_state, learning_rate, discount_factor, epsilon)

class Predator(Agent):
	""" Predator agent, inherits from Agent class """
	def __init__(self, policy, index):
		""" Initializes Predator by calling Agent init """
		Agent.__init__(self, policy, index)

	def __repr__(self):
		""" Represent Predator as X """
		return ' X '	

class Prey(Agent):
	""" Prey Agent, inherits from Agent class """
	def __init__(self, policy, index):
		""" Initializes Prey by calling Agent init """
		Agent.__init__(self, policy, index)

	def __repr__(self):
		""" Represent Prey as O """
		return ' O '		
