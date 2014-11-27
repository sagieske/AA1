import numpy as np
import copy 
import random

class Agent(object):
	""" Agent class, with policy """
	def __init__(self, policy_grid, mc_policy=None, N_policy=None, verbose=0):
		#Store given policy
		self.policy_grid = policy_grid
		self.returns_list = None
		self.N_policy = None
		if(mc_policy is not None):
			self.returns_list = mc_policy
		if(N_policy is not None):
			self.N_policy = N_policy
	
		#Set reward to 0
		self.reward = 0
		self.verbose = verbose

	def set_N_policy(self, N_policy):
		self.N_policy = N_policy
		
	def off_mc(self, visited_pairs, returns, discount_factor):
		amount_rounds = len(visited_pairs)
		for pair in visited_pairs:
			pair_w = 0
			pair_t = self.N_policy.get_N_policy(pair[0])[pair[1]][0]
			for i in range(pair_t, amount_rounds-1):
				k_pair = visited_pairs[i]
				policy = self.N_policy.get_policy(k_pair[0])
				current_prob = self.N_policy.get_e_greedy_policy(policy, epsilon=0.0)[k_pair[1]]
				pair_w+=(1.0/current_prob)
			power = amount_rounds - pair_t - 1
			discounted_cumulative_reward = 10 * (discount_factor**power)
			current_values = self.N_policy.get_N_policy(pair[0])[pair[1]]
			N = current_values[2] + pair_w * discounted_cumulative_reward
			D = current_values[3] + pair_w
			if(D==0):
				D=1
			Q = (N/D)
			self.N_policy.update_values(N,D,Q,pair)
			self.policy_grid.update_Q_values(Q, pair)
			#print self.policy_grid.get_policy(pair[0])
			
	def get_N_policy(self, old_state):
		return self.N_policy.get_N_policy(old_state)

	def get_N_policy_grid(self):
		return self.N_policy

	def update_t_value(self, old_state, t):
		self.N_policy.update_t_value(old_state, t)

	def get_greedy_action(self, old_state):
		return self.policy_grid.get_greedy_action(old_state)

	def get_mc_policy(self):
		return self.returns_list

	def update_returns(self, pairs_list, cumulative_reward, discount_factor):
		len_list = len(pairs_list)
		for i in range(0, len_list):
			power = len_list - i - 1 
			discounted_cumulative_reward = cumulative_reward * (discount_factor**power)
			self.returns_list.update_returns_list(pairs_list[i], discounted_cumulative_reward)

	def update_q_values(self, pairs_list):
		for i in pairs_list:
			return_list = self.returns_list.get_returns_pair(i)
			q_sa = sum(return_list)/len(return_list)
			self.policy_grid.update_q(i, q_sa)

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

	def get_action(self, state, epsilon=0.0, restricted=None):
		#Retrieve an action using the policy for this state in the policy object 
		return self.policy_grid.get_action(state, epsilon, restricted)

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


class Predator(Agent):
	""" Predator agent, inherits from Agent class """
	def __init__(self, policy, mc_policy=None, N_policy=None, verbose=0):
		""" Initializes Predator by calling Agent init """
		Agent.__init__(self, policy, mc_policy, N_policy)

	def __repr__(self):
		""" Represent Predator as X """
		return ' X '	

	def q_learning(self, action, old_state, new_state, learning_rate, discount_factor, epsilon):
		self.policy_grid.q_learning(action, old_state, new_state, learning_rate, discount_factor, epsilon)

	def sarsa(self, action, old_state, new_state, learning_rate, discount_factor, epsilon):
		self.policy_grid.sarsa(action, old_state, new_state, learning_rate, discount_factor, epsilon)

class Prey(Agent):
	""" Prey Agent, inherits from Agent class """
	def __init__(self, policy):
		""" Initializes Prey by calling Agent init """
		Agent.__init__(self, policy)

	def __repr__(self):
		""" Represent Prey as O """
		return ' O '		

	def get_action(self, state, epsilon=0.0, restricted=None):
		return self.policy_grid.get_action_prey(state, restricted)
