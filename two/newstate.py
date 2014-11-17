import random
import math
import sys
import argparse
import numpy as np
import time
from math import ceil, floor
import pdb
from agents_new import Predator, Prey
import helpers
from other_objects import Environment, Policy

class Game:
	def __init__(self, reset=False, prey=None, predator=None, predator_location=[0,0], prey_location=[2,2], verbose=2, grid_size=[11,11]):
		""" Initalize environment and agents """
		#Instantiate environment object with correct size, predator and prey locations
		self.environment = Environment(grid_size, predator_location, prey_location)
		#Create prey if none was given
		if(prey==None):
			prey_policy = Policy(grid_size, prey=True)
			self.prey = Prey(prey_policy)
		else:
			self.prey = prey
			if reset:
				self.environment.place_object('predator', prey_location)
		#Create predator if none was given
		if(predator==None):
			predator_policy = Policy(grid_size, prey=False)
			self.predator = Predator(predator_policy)
		else:
			self.predator = predator
			if reset:
				self.environment.place_object('prey', predator_location)
		#Set level of output
		self.verbose = verbose
		#Print if needed
		if self.verbose > 0:
			self.environment.print_grid()

		print "Episode created with grid size ", grid_size, ", predator at location ", predator_location, ", prey at location ", prey_location

	def get_rounds(self, epsilon, discount_factor, alpha):
		""" Return rounds played """
		#Play rounds until the prey is caught, and return how many were needed
		self.rounds = self.until_caught(epsilon, discount_factor, alpha)
		return self.rounds

	def until_caught(self, epsilon, discount_factor, alpha):
		""" Repeat turns until prey is caught. Returns number of steps until game stopped """
		steps = 0
		caught = 0
		#Runs turns until the prey is caught
		while(caught == 0):
			steps +=1
			#Get the current state
			state = self.environment.get_state()
			#Run turn and see if prey has been caught
			caught = self.turn(state, epsilon, discount_factor, alpha)
			self.predator.update_reward(0)
		#If the prey has been caught, the predator receives a reward of 10
		self.predator.update_reward(10)
		predator_location = self.environment.get_location('predator')
		prey_location = self.environment.get_location('prey')
		distance = [abs(predator_location[0] - prey_location[0]), abs(predator_location[1] - prey_location[1])]
		print 'States: '
		print 'Distance: ', distance
		print 'Predator: ', predator_location[0], ',', predator_location[1]
		print 'Prey: ', prey_location[0], ',', prey_location[1]
		self.environment.print_grid()
		print "Caught prey in " + str(steps) + " rounds!\n=========="
		return steps

	def turn(self, state, epsilon, discount_factor, alpha):
		""" Plays one turn for prey and predator. Choose their action and adjust their state and location accordingly """
		#Get current prey location
		prey_location = self.environment.get_location('prey')
		#Move the predator
		predator_location, predator_action = self.turn_predator(state, epsilon, discount_factor, alpha)
		#If predator moves into the prey, the prey is caught
		same = (predator_location == prey_location)
		self.predator.q_learning(state, predator_action, predator_location, epsilon, discount_factor, alpha, same)
		if(not same):
			#If prey is not caught, move it
			prey_location = self.turn_prey(state, predator_location)
			distance = [abs(predator_location[0] - prey_location[0]), abs(predator_location[1] - prey_location[1])]
			#Print effect of this turn
			if (self.verbose == 1 and same):
				print 'States: '
				print 'Distance: ', distance
				print 'Predator: ', predator_location[0], ',', predator_location[1]
				print 'Prey: ', prey_location[0], ',', prey_location[1]
				self.environment.print_grid()
			elif self.verbose == 2:
				print 'States: '
				print 'Distance: ', distance
				print 'Predator: ', predator_location[0], ',', predator_location[1]
				print 'Prey: ', prey_location[0], ',', prey_location[1]
				self.environment.print_grid()
		#Return caught or not
		return same

	def turn_prey(self, state, predator_location):
		""" Perform turn for prey """
		#Retrieve the action for the prey for this state
		prey_move, action_name = self.prey.get_action(state, predator=False)
		#Turn action into new location
		new_location = self.get_new_location('prey', prey_move)
		#Check if the new location contains the predator, and if so, pick different action
		if new_location == predator_location:
			#Get action, restricted by predator location
			prey_move, action_name = self.prey.get_action(state, restricted=[action_name], predator=False)
			#Turn action into new location
			new_location = self.get_new_location('prey', prey_move)
		#Move the prey to the new location
		self.environment.move_object('prey', new_location)
		return new_location

	def turn_predator(self, state, epsilon, discount_factor, alpha):
		""" Perform turn for predator """
		#Retrieve the action for the predator for this state
		predator_move, action_name = self.predator.get_action(state, epsilon=epsilon, discount_factor=discount_factor, alpha=alpha, predator=True)
		#Turn the action into new location
		new_location = self.get_new_location('predator', predator_move)
		#Move the predator to the new location
		self.environment.move_object('predator', new_location)
		return new_location, action_name

	def get_new_location(self, chosen_object, chosen_move):
		""" Returns new location of an object when performs the chosen move """
		new_location = []
		#Retrieve the agent's position in the grid
		old_location = self.environment.get_location(chosen_object)
		#Get the size of the environment
		environment_size = self.environment.get_size()
		#Wrap edges to make grid toroidal
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

	def wrap_state(self, state, gridsize, encoding):
		""" Wrap states for non-encoding for toroidal grid"""
		# Only wrap for non-encoding
		if not encoding:
			temp_state = state
			state[0] = temp_state[0] % gridsize[0]
			state[1] = temp_state[1] % gridsize[1]
		return state

	def transition(self, old_state, new_state, action):
		""" Calculate transition value from old state to new state """
		old_predator_state = [old_state[0], old_state[1]]
		old_prey_state = [old_state[2], old_state[3]]
		new_predator_state = [new_state[0], new_state[1]]
		new_prey_state = [new_state[2], new_state[3]]
		new_location = self.get_new_state_location(old_predator_state, action)

		# Check if predator can move to its new location given action
		if(new_predator_state == new_location):
			is_allowed = 1
		elif(new_prey_state == old_predator_state):
			is_allowed = 0
		else:
			is_allowed = 0
		# Get probability of prey moving to new prey state
		if(new_prey_state == old_prey_state):
			prey_probability = 0.8
		else:
			prey_probability = 0.05
		return is_allowed * prey_probability

	def predator_transition(self, old_state, new_state, action):
		""" Calculate if predator can move to new state """
		new_location = self.get_new_state_location(old_state, action)
		if(new_state == new_location):
			return 1
		else:
			return 0

	def prey_transition(self, old_state, new_state, action):
		""" Calculate probability of prey moving to new prey state """
		if old_state == new_state:
			return 0.8
		else:
			return 0.05

	def q_value(self, state, action, value_grid, discount_factor, grid_size):
		""" Calculate q value for given state and action"""
		# Initialize value
		i = state[0] 
		j = state[1]
		k = state[2]
		l = state[3]
		# Predator is at same location as prey
		if([i,j] == [k,l]):
			return 0
		q_value = 0
		# Initialize possible new states for predator and prey
		possible_new_predator = [[i,j], [i+1,j], [i,j+1], [i-1,j], [i,j-1]]
		possible_new_prey = [[k,l], [k+1,l], [k,l+1], [k-1,l], [k,l-1]]
		# Loop over possible predator states
		for new_state in possible_new_predator:
			# Check for toroidal grid
			new_state = self.wrap_state(new_state, grid_size, False)
			# Get reward 
			reward = self.reward(state, [new_state[0], new_state[1], k, l], action)
			# Get transition value for predator move
			transition_pred = self.predator_transition([i,j], new_state, action)
			#If the reward is 10, the value of the next state is 0 (because: terminal)
			if (reward == 10 and transition_pred == 1):
				return 10
			#If the action cannot lead to the next state, the entire value will be 0
			else:
				# Loop over all possible new states for prey
				for new_prey in possible_new_prey:
					# Check or toroidal
					new_prey = self.wrap_state(new_prey, grid_size, False)
					# Prey cannot run into predator
					if(new_prey == new_state):
						continue
					# Prey transitions
					if(new_prey == [k,l]):
						transition_prey = 0.8
					else:
						transition_prey = 0.05
					# Add to qvalue
					q_value += transition_prey*transition_pred * discount_factor * value_grid[new_state[0]][new_state[1]][new_prey[0]][new_prey[1]]
		return q_value

def run_episodes(policy, predator, grid_size, N, epsilon, discount_factor, alpha):
	""" Run N episodes and compute average """
	total_rounds = 0
	rounds_list = []
	for x in range(0, N):
		#Initialize episode
		game = Game(grid_size=grid_size)
		#Run episode until prey is caught
		current_rounds = game.get_rounds(epsilon, discount_factor, alpha)
		#Add rounds needed in this episode to total_rounds
		total_rounds += current_rounds
		#Add rounds needed in this episode to the list of rounds
		rounds_list.append(current_rounds)
	#Compute average number of rounds needed
	average_rounds = float(total_rounds)/N
	#Compute list of variances
	var_list = [(x-average_rounds)**2 for x in rounds_list]
	#Compute average variance
	variance = float(sum(var_list)/len(var_list))
	#Compute standard deviation for N rounds
	standard_deviation = math.sqrt(variance)
	print "Average rounds needed over ", N, " episodes: ", average_rounds
	print "Standard deviation: ", standard_deviation	

if __name__ == "__main__":
	#Command line arguments
	parser = argparse.ArgumentParser(description="Run simulation")
	parser.add_argument('-runs', metavar='How many simulations should be run?', type=int)
	parser.add_argument('-discount', metavar='Specify the size of the discount factor for value iteration.', type=float)
	parser.add_argument('-verbose', metavar='Verbose level of game. 0: no grids/states, 1: only start and end, 2: all', type=int)
	parser.add_argument('-size', metavar='Size of the grid.', type=int)
	parser.add_argument('-epsilon', metavar='Epsilon for e-greedy', type=float)
	parser.add_argument('-alpha', metavar='Learning rate for Q-learning', type=float)
	args = parser.parse_args()

	N = 100
	discount_factor = 0.1
	size = 11
	epsilon = 0.1
	alpha = 0.1
	if(vars(args)['runs'] is not None):
		N = vars(args)['runs']
	if(vars(args)['discount'] is not None):
		discount_factor = vars(args)['discount']
	if(vars(args)['size'] is not None):
		size = vars(args)['size']
	if(vars(args)['epsilon'] is not None):
		epsilon = vars(args)['epsilon']
	if(vars(args)['alpha'] is not None):
		alpha = vars(args)['alpha']
	if(vars(args)['verbose'] is not None):
		verbose = vars(args)['verbose']
	else:
		verbose = 2


	run_episodes("policy", "predator", [size,size], N, epsilon, discount_factor, alpha)