import numpy as np

class Environment:

	def __init__(self, size, predator_location, prey_location):
		"""Initialize environment of given size"""
		self.size = size
		self.grid = [[ ' ' for i in range(0, size[0])] for y in range(0, size[1])]
		i = predator_location[0]
		j = predator_location[1]
		k = prey_location[0]
		l = prey_location[1]
		self.grid[i][j] = 'X'
		self.grid[k][l] = 'O'
		self.predator_location = predator_location
		self.prey_location = prey_location

	def print_grid(self):
		""" Print the environment"""
		print "=========="
		for row in self.grid:
			print row
		print "=========="

	def place_object(self, grid_object, new_location):
		""" Place an object at a given location in the environment"""
		self.grid[new_location[0]][new_location[1]] = grid_object

	def move_object(self, object_name, new_location):
		if(object_name == 'predator'):
			old_location = self.predator_location
			self.predator_location = new_location
			self.grid[old_location[0]][old_location[1]] = ' '
			self.grid[new_location[0]][new_location[1]] = 'X'
		elif(object_name == 'prey'):
			old_location = self.prey_location
			self.prey_location = new_location
			self.grid[old_location[0]][old_location[1]] = ' '
			self.grid[new_location[0]][new_location[1]] = 'O'			

	def get_size(self):
		""" Return environment size"""
		return self.size

	def get_state(self):
		return [self.predator_location[0], self.predator_location[1], self.prey_location[0], self.prey_location[1]]

	def get_location(self, object_name):
		if(object_name == 'predator'):
			return self.predator_location
		elif(object_name == 'prey'):
			return self.prey_location

class Policy:

	def __init__(self, grid_size, policy_grid=None, prey=False):
		self.grid_size = grid_size
		if prey==False:
			self.policy = {'North':0.2, 'East':0.2, 'South':0.2, 'West':0.2, 'Wait':0.2}
		else:
			self.policy = {'North':0.05, 'East':0.05, 'South':0.05, 'West':0.05, 'Wait':0.8}
		if(policy_grid is not None):
			self.policy_grid = policy_grid
		else:
			self.policy_grid = [[[[self.policy for i in range(0, self.grid_size[1])] for j in range(0, self.grid_size[0])] for k in range(0, self.grid_size[1])] for l in range(0, self.grid_size[0])]
		self.actions = {'North': [-1,0], 'East': [0,1], 'South': [1,0], 'West': [0,-1], 'Wait': [0,0]}

	def get_policy(self, state):
		i = state[0]
		j = state[1]
		k = state[2]
		l = state[3]
		return self.policy_grid[i][j][k][l]

	def get_action(self, state):
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
