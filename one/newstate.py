import random
import math
import sys
import argparse
import numpy as np
import time
from math import ceil, floor
import pdb
from agents import Predator, Prey
import helpers

class Game:
	def __init__(self, reset=False, prey=None, predator=None, prey_location=[5,5], predator_location=[0,0], verbose=2):
		""" Initalize environment and agents """
		# Initialize environment
		self.environment = Environment()

		# Initialize prey and predators
		prey_predator_distance = helpers.xy_distance(predator_location, prey_location, self.environment.get_size())
		print prey_predator_distance
		if(prey==None):
			self.prey = Prey(prey_location)
		else:
			self.prey = prey
			# Reset to start position
			if reset:
				self.prey.set_location(prey_location)
		if(predator==None):
			self.predator = Predator(predator_location, prey_predator_distance)
		else:
			self.predator = predator
			# Reset to start position and reset award value
			if reset:
				self.predator.set_location(predator_location)
				#self.predator.reset_reward()

		# Specify level of verbose output
		self.verbose = verbose

		#Place prey and predator on board
		self.environment.place_object(self.prey, self.prey.get_location())
		self.environment.place_object(self.predator, self.predator.get_location())
		if self.verbose > 0:
			self.environment.print_grid()

	def value_iteration(self, grid_size, epsilon, discount_factor):
		x_length = grid_size[0]
		y_length = grid_size[1]
		converged = False
		v_grid = np.zeros((x_length, y_length, x_length, y_length))
		temp_grid = np.zeros((x_length, y_length, x_length, y_length))
		delta_grid = np.zeros((x_length, y_length, x_length, y_length))
		actions =  self.predator.get_policy().keys()
		print "Actions: ", actions
		print "grid: ", grid_size
		loop = 0
		while(not converged):
			loop +=1
			for i in range(0, x_length):
				for j in range(0, y_length):
					for k in range(0, x_length):
						for l in range(0, y_length):
							highest_q = 0
							best_action = ""
							for action in actions:
								q_value = self.q_value([i,j,k,l], action, temp_grid, discount_factor, grid_size)
								if q_value > highest_q:
									highest_q = q_value
							v_grid[i][j][k][l] = highest_q

							#print "q_value: ", highest_q

			delta_grid = abs(np.array(v_grid) - np.array(temp_grid))
			helpers.pretty_print(delta_grid[:][:][5][5], label=['delta'])
			helpers.pretty_print(temp_grid[:][:][5][5], label = ['temp'])
			helpers.pretty_print(v_grid[:][:][5][5], label=['v'])
			temp_grid = v_grid
			v_grid = np.zeros((x_length, y_length, x_length, y_length))
			
			#v_grid = np.zeros((x_length, y_length, x_length, y_length))
			delta = np.amax(delta_grid)
			print delta
			print "LOOP: ", loop, "\n"
			
			if(delta < epsilon* (1-discount_factor)/discount_factor):
				converged = True

	def get_new_state_location(self, old_location, action):
		""" Returns new state given old state and an action (no object is used) """
		new_location = []
		chosen_move = self.predator.get_transformation(action)
		environment_size = self.environment.get_size()
		# division by modulo makes board toroidal:
		new_location.append((old_location[0] + chosen_move[0]) % environment_size[0])
		new_location.append((old_location[1] + chosen_move[1]) % environment_size[1])
		return new_location

	def wrap_state(self, state, gridsize, encoding):
		""" Wrap states for non-encoding for toroidal grid"""
		# Only wrap for non-encoding
		if not encoding:
			temp_state = state
			state[0] = temp_state[0] % gridsize[0]
			state[1] = temp_state[1] % gridsize[1]
		return state

	def reward(self, old_state, new_state, action):
		if old_state[0] == old_state[2] and old_state[1] == old_state[3]:
			return 0
		elif new_state[0] == new_state[2] and new_state[1] == new_state[3]:
			return 10
		else:
			return 0

	def transition(self, old_state, new_state, action):
		old_predator_state = [old_state[0], old_state[1]]
		old_prey_state = [old_state[2], old_state[3]]
		new_predator_state = [new_state[0], new_state[1]]
		new_prey_state = [new_state[2], new_state[3]]
		new_location = self.get_new_state_location(old_predator_state, action)

		if(new_predator_state == new_location):
			is_allowed = 1
		elif(new_prey_state == old_predator_state):
			is_allowed = 0
		else:
			is_allowed = 0
		if(new_prey_state == old_prey_state):
			prey_probability = 0.8
		else:
			prey_probability = 0.05
		return is_allowed * prey_probability

	def predator_transition(self, old_state, new_state, action):
		new_location = self.get_new_state_location(old_state, action)
		if(new_state == new_location):
			return 1
		else:
			return 0

	def prey_transition(self, old_state, new_state, action):
		if old_state == new_state:
			return 0.8
		else:
			return 0.05

	def q_value(self, state, action, value_grid, discount_factor, grid_size):
		i = state[0] 
		j = state[1]
		k = state[2]
		l = state[3]
		if([i,j] == [k,l]):
			return 0
		q_value = 0
		possible_new_predator = [[i,j], [i+1,j], [i,j+1], [i-1,j], [i,j-1]]
		possible_new_prey = [[k,l], [k+1,l], [k,l+1], [k-1,l], [k,l-1]]
		for new_state in possible_new_predator:
			new_state = self.wrap_state(new_state, grid_size, False)
			reward = self.reward(state, [new_state[0], new_state[1], k, l], action)
			transition_pred = self.predator_transition([i,j], new_state, action)
			#If the reward is 10, the value of the next state is 0 (because: terminal)
			if (reward == 10 and transition_pred == 1):
				return 10
			#If the action cannot lead to the next state, the entire value will be 0
			else:
				for new_prey in possible_new_prey:
					new_prey = self.wrap_state(new_prey, grid_size, False)
					if(new_prey == new_state):
						continue
					if(new_state == [k,l]):
						transition_prey = 0.8
					else:
						transition_prey = 0.05
					q_value += transition_prey*transition_pred * discount_factor * value_grid[new_state[0]][new_state[1]][new_prey[0]][new_prey[1]]
		return q_value

 

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

    
	def get_new_state_location(self, old_location, action):
		""" Returns new state given old state and an action (no object is used) """
		new_location = []
		chosen_move = self.predator.get_transformation(action)
		environment_size = self.environment.get_size()
		# division by modulo makes board toroidal:
		new_location.append((old_location[0] + chosen_move[0]) % environment_size[0])
		new_location.append((old_location[1] + chosen_move[1]) % environment_size[1])
		return new_location

		
	def get_action(self, old_state, new_state):
		""" Returns the action which should be used to obtain a specific new state from the old one """
		actions = self.predator.get_policy().keys()
		
		# Loop through the actions and find which one yields the desired new_state
		for action in actions:
		      new_location = []
		      chosen_move = self.predator.get_transformation(action)
		      environment_size = self.environment.get_size()
		      # division by modulo makes board toroidal:
		      new_location.append((old_state[0] + chosen_move[0]) % environment_size[0])
		      new_location.append((old_state[1] + chosen_move[1]) % environment_size[1])
		      
		      # Return the action which yields the new state
		      if new_location == new_state:
		          return action
		          


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

	N = 100
	discount_factor = 0.8
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
	predator = Predator([5,5], [5,5])
	game = Game(reset=True, prey=prey, predator=predator, verbose=verbose)

	goal_state = [5,5]
	grid_size = [11,11]
	game.value_iteration(grid_size, 0.00001, 0.9)

