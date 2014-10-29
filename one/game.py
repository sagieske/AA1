import random
import math
import sys
import argparse
import numpy as np
#from helpers import common_entries

'''Predator class, with policy'''
class Predator:
	def __init__(self, location):
		self.policy = {'North':0.2, 'East':0.2, 'South':0.2, 'West':0.2, 'Wait':0.2}
		self.actions = {'North': [-1,0], 'East': [0,1], 'South': [1,0],'West': [0,-1], 'Wait':[0,0]}
		self.location = location
		self.state = "Predator(" + str(self.location[0]) + "," + str(self.location[1]) + ")"
		self.reward = 0

	def __repr__(self):
		""" Represent Predator as X """
		return ' X '

	def action(self):
		""" Choose an action and turn it into a move """
		chosen_action = self.pick_action()
		chosen_move = self.actions[chosen_action]
		return chosen_move, chosen_action

	def pick_action(self):
		""" Use the probabilities in the policy to pick a move """
		# Split policy dictionary in list of keys and list of values
		action_name, policy = zip(*self.policy.items())
		# Get choice using probability distribution
		choice_index = np.random.choice(list(action_name), 1, list(policy))[0]
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

	def update_reward(self, reward):
		""" Add reward gained on time step to total reward """
		self.reward += reward

	def reset_reward(self):
		""" Reset reward to inital value """
		self.reward = 0

	def get_reward(self):
		""" Get collected reward for predator """
		return self.reward

	def get_policy(self):
		""" Return the predator's policy """
		return self.policy

'''Prey class, with policy'''
class Prey:
	def __init__(self, location):
		""" Initialize Prey with standard policy """
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
		action_name, policy = zip(*self.policy.items())
		# Get choice using probability distribution
		choice_index = np.random.choice(list(action_name), 1, list(policy))[0]
		return choice_index

	def pick_action_restricted(self, blocked_moves):
		""" Use the probabilities in the policy to pick a move but can not perform blocked move """
		# Temporary policy list
		temp_policy = self.policy
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

class Game:
	def __init__(self, reset=False, prey=None, predator=None, prey_location=[5,5], predator_location=[0,0], verbose=2):
		""" Initalize environment and agents """
		# Initialize prey and predators
		if(prey==None):
			self.prey = Prey(prey_location)
		else:
			self.prey = prey
			# Reset to start position
			if reset:
				self.prey.set_location(prey_location)
		if(predator==None):
			self.predator = Predator(predator_location)
		else:
			self.predator = predator
			# Reset to start position and reset award value
			if reset:
				self.predator.set_location(predator_location)
				#self.predator.reset_reward()
		# Initialize environment
		self.environment = Environment()
		# Specify level of verbose output
		self.verbose = verbose

		#Place prey and predator on board
		self.environment.place_object(self.prey, self.prey.get_location())
		self.environment.place_object(self.predator, self.predator.get_location())
		if self.verbose > 0:
			self.environment.print_grid()

	def value_iteration(self, discount_factor, loops, start_location_prey=[5,5], gridsize=[11,11]):
		""" Performs value iteration """
		#Initialize parameters
		x_size = gridsize[0]
		y_size = gridsize[1]
		convergence = False

		# Initialize grids
		value_grid = np.zeros((x_size, y_size))
		new_grid = np.zeros((x_size, y_size))
		delta_grid = np.zeros((x_size,y_size))

		# Set goal state reward
		value_grid[start_location_prey[0]][start_location_prey[1]] = 10

		count = 0
		# Continue value iteration until convergence has taken place
		while(not convergence):
			# Get all nonzero indices from value_grid
			nonzero_indices = np.transpose(np.nonzero(value_grid))

			# Calculate surrounding values for non zero elements
			for item in nonzero_indices:
				# Set indices
				i  = item[0]
				j = item[1]
				# Get surrounding states
				possible_new_states = [[i,j], [i+1,j], [i-1,j], [i,j+1], [i,j-1]]
				for new_state in possible_new_states:
					temp_state = new_state
					#Check for toroidal wrap
					new_state[0] = temp_state[0] % x_size
					new_state[1] = temp_state[1] % y_size
					# Get value for state
					value = self.get_value(new_state, start_location_prey, discount_factor, [x_size, y_size], value_grid)
					# Update grid
					new_grid[new_state[0]][new_state[1]] = value

			# Get delta between old and new grid
			delta_grid = abs(np.array(new_grid) - np.array(value_grid))

			# Update grids for next round
			value_grid = new_grid
			new_grid = np.zeros((x_size,y_size))

			self.pretty_print(value_grid)
   			print "=====val====="
			for row in value_grid:
   				pretty_row =["%.6f" % v for v in row]
   				print pretty_row
   			print "=========="

			# Get maximum difference between grids
			delta = np.amax(delta_grid)

			count+=1
			# Check for convergence
			if delta < 0.0001:
				convergence = True
				print "Converged! Counter is at: %i" %(count)

	def pretty_print(self, matrix):
		for row in matrix:
			pretty_row = [ v*1000 for v in row]
			print pretty_row


   	def get_value(self, state, goal_state, discount_factor, grid_size, value_grid):
   		if(state == goal_state):
   			return 10
   		else:
   			i = state[0]
	   		j = state[1]
	   		[x_size, y_size] = grid_size
	   		actions =  self.predator.get_policy().iteritems()
			action_values = []
			new_states = [[i,j], [i+1,j], [i-1,j], [i,j+1], [i,j-1]]
			for action in actions:
				prob_sum = 0
				for new_state in new_states:
							#Back up old state
					temp_state = new_state
							#Check for toroidal wrap
					new_state[0] = temp_state[0] % x_size
					new_state[1] = temp_state[1] % y_size
							#Compute transition value from s to s'
					transition_value = self.transition(state, new_state, goal_state, action[0])
							#Compute reward from s to s'
					reward_value = self.reward_function(state, new_state, goal_state, action[0])
							#Add this to the sum of state probabilities
					prob_sum += transition_value * (reward_value + discount_factor * value_grid[new_state[0]][new_state[1]])
						#Append sum of state probabilities for this action times probability for this action to the action list
				action_values.append(prob_sum*action[1])
					#The value for i,j is the max of all action_values
			value = max(action_values)
			return value


	def transition(self, old_state, new_state, goal_state, action):
		#If we're staying in the same place with a non-waiting action, the prob is 0
		if old_state == new_state and action != 'Wait':
			return 0
		#If we're moving while using the wait action, the prob is 0
		elif old_state != new_state and action == 'Wait':
			return 0
		#All other actions have transition probability of 1
		else:
			return 1

	def reward_function(self, old_state, new_state, goal_state, action):
		#All states have a reward of 0, except the terminal state
		return 0

	def get_rounds(self):
		""" Return rounds played """
		self.rounds = self.until_caught()
		return self.rounds

	def until_caught(self):
		""" Repeat turns until prey is caught. Returns number of steps until game stopped """
		steps = 0
		caught = 0
		while(caught == 0):
			steps +=1
			caught = self.turn()
			self.predator.update_reward(0)
		self.predator.update_reward(10)
		print "Caught prey in " + str(steps) + " rounds!\n=========="
		return steps

	def turn(self):
		""" Plays one turn for prey and predator. Choose their action and adjust their state and location accordingly """
		# Play one turn prey
		self.turn_prey()
		# Play one turn predator
		self.turn_predator()

		#Check if prey is caught
		same = (self.predator.get_location() == self.prey.get_location())

		# Only print grid or show prey & predator states if verbose level is 1 or 2 
		if (self.verbose == 1 and same):
			self.environment.print_grid()
			print "States: "
			print self.predator.get_state()
			print self.prey.get_state()
			# Always print grid at verbose level 2
		elif self.verbose == 2:
			self.environment.print_grid()
			print "States: "
			print self.predator.get_state()
			print self.prey.get_state()

		return same

	def turn_prey(self):
		""" Perform turn for prey """
		#Remove prey from old location
		self.environment.remove(self.prey.get_location())
		#Get action for prey
		prey_move, action_name = self.prey.action()
		#Get new location for prey
		new_prey_location = self.get_new_location(self.prey, prey_move)
		#Check if the prey is not stepping on the predator
		if new_prey_location == self.predator.get_location():
			prey_move,action_name = self.prey.action(restricted=[action_name])
			new_prey_location = self.get_new_location(self.prey, prey_move)
			#print "Prey almost stepped on predator! Performed action: %s" %(action_name)
			##If it is, make it wait (hide) instead
			#new_prey_location = self.prey.get_location()
			"Prey almost stepped on predator! It went to hide in the bushes instead."
		#Move prey to new location
		self.environment.place_object(self.prey, new_prey_location)	
		#Update prey's location in its own knowledge
		self.prey.set_location(new_prey_location)

	def turn_predator(self):
		""" Perform turn for predator """
		#Remove predator from old location
		self.environment.remove(self.predator.get_location())
		#Get action for predator
		predator_move,action_name = self.predator.action()
		#Get new location for predator
		new_predator_location = self.get_new_location(self.predator, predator_move)
		#Move predator to new location
		self.environment.place_object(self.predator, new_predator_location)	
		#Update predator's location in its own knowledge
		self.predator.set_location(new_predator_location)

	def get_new_location(self, chosen_object, chosen_move):
		""" Returns new location of an object when performs the chosen move """
		new_location = []
		old_location = chosen_object.get_location()
		environment_size = self.environment.get_size()
		# division by modulo makes board toroidal:
		new_location.append((old_location[0] + chosen_move[0]) % environment_size[0])
		new_location.append((old_location[1] + chosen_move[1]) % environment_size[1])
		return new_location

class Environment:

	def __init__(self, size=[11,11]):
		"""Initialize environment of given size"""
		self.size = size
		self.grid = [[ ' ' for i in range(0, size[0])] for y in range(0, size[1])]

	def print_grid(self):
		""" Print the environment"""
		print "=========="
		for row in self.grid:
			print row
		print "=========="

	def place_object(self, grid_object, new_location):
		""" Place an object at a given location in the environment"""
		self.grid[new_location[0]][new_location[1]] = grid_object

	def remove(self, location):
		""" Remove object on given location """
		self.grid[location[0]][location[1]] = ' '

	def get_size(self):
		""" Return environment size"""
		return self.size

if __name__ == "__main__":
	#Command line arguments
	parser = argparse.ArgumentParser(description="Run simulation")
	parser.add_argument('-runs', metavar='How many simulations should be run?', type=int)
	parser.add_argument('-discount', metavar='Specify the size of the discount factor for value iteration.', type=float)
	parser.add_argument('-loops', metavar='Specify the amount of loops to test value iteration on.', type=int)
	parser.add_argument('-verbose', metavar='Verbose level of game. 0: no grids/states, 1: only start and end, 2: all', type=int)
	args = parser.parse_args()

	N = 1
	discount_factor = 0.9
	loops = 3
	if(vars(args)['runs'] is not None):
		N = vars(args)['runs']
	if(vars(args)['discount'] is not None):
		discount_factor = vars(args)['discount']
	if(vars(args)['loops'] is not None):
		loops = vars(args)['loops']
	if(vars(args)['verbose'] is not None):
		verbose = vars(args)['verbose']
	else:
		verbose = 2

	count = 0
	count_list = []
	#Initialize re-usable prey and predator objects
	prey = Prey([0,0])
	predator = Predator([5,5])
	#Run N games
	for x in range(0, N):
		# Start game and put prey and predator at initial starting position
		game = Game(reset=True, prey=prey, predator=predator, verbose=verbose)
		rounds = game.get_rounds()
		count += rounds
		count_list.append(rounds)
		print 'Cumulative reward for ' + str(x+1) + ' games: ' + str(predator.get_reward())
	#Calculate average steps needed to catch prey
	average = float(count/N)
	#Calculate corresponding standard deviation
	var_list = [(x-average)**2 for x in count_list]
	variance = float(sum(var_list)/len(var_list))
	standard_deviation = math.sqrt(variance)
	print "Average amount of time steps needed before catch over " + str(N) + " rounds is " + str(average) + ", standard deviation is " + str(standard_deviation)
	#Perform value_iteration over the policy
	game.value_iteration(discount_factor, loops)
