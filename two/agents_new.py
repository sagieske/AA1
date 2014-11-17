import numpy as np
import copy 
import random

class Agent(object):
	""" Agent class, with policy """
	def __init__(self, policy_grid):
		#Store given policy
		self.policy_grid = policy_grid
		#Set reward to 0
		self.reward = 0

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
		""" Return the agent's policy """
		#Get indices to retrieve policy
		x_distance = state[0]
		y_distance = state[1]
		return self.policy[x_distance][y_distance]

	def get_policy_grid(self):
		""" Return policy grid for agent """
		return self.policy

	def set_policy_grid(self, policy_grid):
		""" Set policy grid for agent """
		self.policy = policy_grid

	def get_action(self, state, restricted=None, epsilon=0.0):
		"""Retrieve an action using the policy for this state in the policy object """
		return self.policy_grid.get_action(state, restricted=restricted, epsilon=epsilon)		


class Predator(Agent):
	""" Predator agent, inherits from Agent class """
	def __init__(self, policy):
		""" Initializes Predator by calling Agent init """
		Agent.__init__(self, policy)

	def __repr__(self):
		""" Represent Predator as X """
		return ' X '		

class Prey(Agent):
	""" Prey Agent, inherits from Agent class """
	def __init__(self, policy):
		""" Initializes Prey by calling Agent init """
		Agent.__init__(self, policy)

	def __repr__(self):
		""" Represent Prey as O """
		return ' O '		
