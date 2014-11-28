import random
import math
import sys
import argparse
import numpy as np
import time
from math import ceil, floor
import pdb
from agents_new2 import Predator, Prey
import helpers
from other_objects2 import Environment, Policy
import matplotlib.pyplot as plt

class Game:
	def __init__(self, reset=False, prey=None, predator=None, predator_location=[0,0], prey_location=[5,5], softmax=False, verbose=2, grid_size=[11,11], learning_type='Q-learning'):
		""" Initalize environment and agents """
		print "LEARNING: ", learning_type, "\n\n"
		self.learning_type = learning_type
		self.visited_pairs = []
		#print 'learning type: ', self.learning_type
		
		if(self.learning_type == "ONMC"):
			mc_policy = Policy(grid_size, prey=False, softmax=softmax, verbose=verbose, mc=True)
			N_policy = None
		elif(self.learning_type == "OFFMC" and predator is None):
			#print 'getting N_policy!'
			N_policy = Policy(grid_size, prey=False, softmax=softmax, verbose=verbose, mc=False, off_policy=True)
			mc_policy = None
			#print "npol: ",N_policy
		else:
			if(self.learning_type != "OFFMC"):
				#print 'learning type not offmc'
				#print 'Setting N policy to none!'
				N_policy = None
			mc_policy = None
		#Instantiate environment object with correct size, predator and prey locations
		self.environment = Environment(grid_size, predator_location, prey_location)
		#Create prey if none was given
		if(prey==None):
			prey_policy = Policy(grid_size, prey=True, softmax=softmax, verbose=verbose)
			self.prey = Prey(prey_policy)
		else:
			self.prey = prey
			if reset:
				self.environment.place_object('predator', prey_location)
		#Create predator if none was given
		if(predator==None):
			predator_policy = Policy(grid_size, prey=False, softmax=softmax, verbose=verbose)
			if(self.learning_type != "OFFMC"):
				self.predator = Predator(predator_policy, mc_policy)
			else:
				self.predator = Predator(predator_policy, mc_policy=None, N_policy=N_policy)
				#print "cassandra is echt kut: ", N_policy
		else:
			self.predator = predator
			#if (self.learning_type == "OFFMC"):
				#print 'setting N_policy!'
			#	self.predator.set_N_policy(N_policy)
			if reset:
				self.environment.place_object('prey', predator_location)
		#Set level of output
		self.verbose = verbose
		#Print if needed
		if self.verbose > 0:
			self.environment.print_grid()

		print "Episode created with grid size ", grid_size, ", predator at location ", predator_location, ", prey at location ", prey_location

	def get_rounds(self, learning_rate, discount_factor, epsilon):
		""" Return rounds played """
		#Play rounds until the prey is caught, and return how many were needed
		self.rounds = self.until_caught(learning_rate, discount_factor, epsilon)
		return self.rounds

	def until_caught(self, learning_rate, discount_factor, epsilon):
		""" Repeat turns until prey is caught. Returns number of steps until game stopped """
		steps = 0
		caught = 0
		#Runs turns until the prey is caught
		action=None
		while(caught == 0):
			steps +=1
			#Get the current state
			state = self.environment.get_state()
			#Run turn and see if prey has been caught
			caught, action = self.turn(state, learning_rate, discount_factor, epsilon, steps, action)
			self.predator.update_reward(0)
			newstate = self.environment.get_state()
		#If the prey has been caught, the predator receives a reward of 10
		self.predator.update_reward(10)
		if(learning_type == "OFFMC"):
			returns = 10
			self.predator.off_mc(self.visited_pairs, returns, discount_factor)
			#if(self.learning_type=="OFFMC"):
			#	print "state: ", newstate
			#	print self.predator.get_policy_grid().get_policy(newstate)
		reward = 10
		
		print "Caught prey in " + str(steps) + " rounds!\n=========="
		return reward, self.visited_pairs, steps, self.predator.get_policy_grid(), self.predator.get_mc_policy(), self.predator.get_N_policy_grid()

	def relative_xy(self, location1, location2):
		""" Get relative(shortest) distance between two locations using the toroidal property"""
		# Get grid size of the game
		grid_size = self.environment.get_size()
		# Get relative distance to prey using toroidal property
		distance_x = min(abs(state_prey[0] - state_predator[0]), abs(grid_size[0] - abs(state_prey[0] - state_predator[0])))
		distance_y = min(abs(state_prey[1] - state_predator[1]), abs(grid_size[1] - abs(state_prey[1] - state_predator[1])))
		return [distance_x, distance_y]

	def turn(self, old_state, learning_rate, discount_factor, epsilon, steps, action=None):
		""" Plays one turn for prey and predator. Choose their action and adjust their state and location accordingly """
		#Get current prey location
		prey_location = [old_state[2], old_state[3]]
		#Move the predator
		if(action is not None):
			predator_location = self.get_new_location('predator', action[1])
			predator_action = action[0]
		else:
			predator_location, predator_action = self.turn_predator(old_state)
		new_state = [predator_location[0], predator_location[1], prey_location[0], prey_location[1]]
		if self.verbose > 0:
			print "predator_location: ", predator_location, " prey_location: ", prey_location, " old state: ", old_state, " new state: ", new_state
		#If predator moves into the prey, the prey is caught
		same = (predator_location == prey_location)

		if(self.learning_type == 'Q-learning'):
			self.predator.q_learning(predator_action, old_state, new_state, learning_rate, discount_factor, epsilon)
		elif(self.learning_type == 'Sarsa'):
			action = self.predator.sarsa(predator_action, old_state, new_state, learning_rate, discount_factor, epsilon)
		elif(self.learning_type == 'ONMC'):
			self.visited_pairs.append((old_state, predator_action))
		elif(self.learning_type == 'OFFMC'):
			self.visited_pairs.append((old_state, predator_action))
			greedy_action = self.predator.get_greedy_action(old_state)
			if(greedy_action != predator_action):
				self.predator.update_t_value(old_state, steps)
			#print self.predator.get_N_policy_grid().get_N_policy(old_state)
			#print old_state
		#else:
		#	print self.predator.get_policy_grid().get_policy(old_state)
		#	print predator_action

		if(not same):
			#If prey is not caught, move it
			prey_location = self.turn_prey(old_state, predator_location, epsilon)
			#Print effect of this turn
			if (self.verbose == 1 and same):
				print 'States: '
				print 'Predator: ', predator_location[0], ',', predator_location[1]
				print 'Prey: ', prey_location[0], ',', prey_location[1]
				self.environment.print_grid()
			elif self.verbose == 2:
				print 'States: '
				print 'Predator: ', predator_location[0], ',', predator_location[1]
				print 'Prey: ', prey_location[0], ',', prey_location[1]
				self.environment.print_grid()
		#Return caught or not
		return same, action

	def turn_prey(self, state, predator_location, epsilon):
		""" Perform turn for prey """
		#Retrieve the action for the prey for this state
		prey_move, action_name = self.prey.get_action(state, epsilon)
		#Turn action into new location
		new_location = self.get_new_location('prey', prey_move)
		#Check if the new location contains the predator, and if so, pick different action
		if new_location == predator_location:
			print "PREDATOR IS HERE YO - DONT GO THERE (we say to the prey)"
			#Get action, restricted by predator location
			prey_move, action_name = self.prey.get_action(state, epsilon, restricted=[action_name])
			#Turn action into new location
			new_prey_location = self.get_new_location('prey', prey_move)
		#Move the prey to the new location
		self.environment.move_object('prey', new_location)
		return new_location

	def turn_predator(self, state):
		""" Perform turn for predator """
		#Retrieve the action for the predator for this state
		if(self.learning_type == "OFFMC"):
			predator_move, action_name = self.predator.get_action(state, epsilon=1.0)
		elif(self.learning_type == "None"):
			predator_move, action_name = self.predator.get_action(state, epsilon=0.0)
			#print "action: ", action_name
			#print "policy: ", self.predator.get_policy_grid().get_policy(state)
			#print "state: ", state
		else:
			predator_move, action_name = self.predator.get_action(state, epsilon)
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

	def reward(self, old_state, new_state, action):
		""" Calculate reward for transition from old state to new state"""
		if old_state[0] == old_state[2] and old_state[1] == old_state[3]:
			return 0
		elif new_state[0] == new_state[2] and new_state[1] == new_state[3]:
			return 10
		else:
			return 0

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
def run_episodes(grid_size, N, learning_rate, discount_factor, epsilon, softmax=False, verbose=0, learning_type='Q-learning'):
	""" Run N episodes and compute average """
	#print 'in run episodes!'
	total_rounds = 0
	rounds_list = []
	policy_grid = Policy(grid_size)
	N_policy = Policy(grid_size,off_policy=True)
	predator = Predator(policy_grid, N_policy=N_policy)
	#game = Game(predator=predator,grid_size=grid_size, softmax=softmax, verbose=verbose, learning_type=learning_type)
	average_list = []
	counter=0
	current_rounds=0
	for x in range(0, N):
		#print 'learning type 1: ', this_learning_type
		#Run episode until prey is caught
		print "current_rounds: ", current_rounds
		game_learning = Game(grid_size=grid_size, predator=predator, softmax=softmax, verbose=verbose, learning_type=learning_type)
		reward, visited_pairs, irrel_rounds, policy_grid, mc_policy, N_policy = game_learning.get_rounds(learning_rate, discount_factor, epsilon)
		predator = Predator(policy_grid)

		if(learning_type == 'ONMC'):
			predator.update_returns(visited_pairs, reward, discount_factor)
			predator.update_q_values(visited_pairs)
		#Initialize episode
		print "Round ", x
		game_testing = Game(grid_size=grid_size, predator=predator, softmax=softmax, verbose=verbose, learning_type="None")
		reward, visited_pairs, current_rounds, policy_grid, mc_policy, irrel_N_policy = game_testing.get_rounds(learning_rate, discount_factor, epsilon=0.0)
		predator = Predator(policy_grid, N_policy=N_policy)
		#Add rounds needed in this episode to total_rounds
		total_rounds += current_rounds
		#Add rounds needed in this episode to the list of rounds
		rounds_list.append(current_rounds)
		counter+=1
		if(counter == 100):
			average_rounds = float(total_rounds)/100
			average_list.append(average_rounds)
			total_rounds = 0
			counter= 0
	print "rounds list: ", rounds_list
	print "average_list: ", average_list
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
	return average_list


if __name__ == "__main__":
	#Command line arguments
	parser = argparse.ArgumentParser(description="Run simulation")
	parser.add_argument('-runs', metavar='How many simulations should be run?', type=int)
	parser.add_argument('-discount', metavar='Specify the size of the discount factor for value iteration.', type=float)
	parser.add_argument('-verbose', metavar='Verbose level of game. 0: no grids/states, 1: only start and end, 2: all', type=int)
	parser.add_argument('-learning_rate', metavar='Specify value of learning rate', type=float)
	parser.add_argument('-epsilon', metavar='Specify value of epsilon', type=float)
	parser.add_argument('-grid_size', metavar='Specify grid size', type=int)
	parser.add_argument('-learning_type', metavar='Specify learning type', type=str)
	parser.add_argument('-softmax', metavar='Use softmax', type=str)
	args = parser.parse_args()

	N = 100
	discount_factor = 0.8
	learning_rate = 0.5
	epsilon = 0.1
	grid_size = 11
	softmax = False	
	learning_type = "Q-learning"
	if(vars(args)['runs'] is not None):
		N = vars(args)['runs']
	if(vars(args)['learning_type'] is not None):
		learning_type = vars(args)['learning_type']
	if(vars(args)['runs'] is not None):
		N = vars(args)['runs']
	if(vars(args)['softmax'] is not None):
		if(vars(args)['softmax'] == "yes"):
			softmax = True
	if(vars(args)['grid_size'] is not None):
		grid_size = vars(args)['grid_size']
	if(vars(args)['discount'] is not None):
		discount_factor = vars(args)['discount']
	if(vars(args)['learning_rate'] is not None):
		learning_rate = vars(args)['learning_rate']
	if(vars(args)['epsilon'] is not None):
		epsilon = vars(args)['epsilon']
	if(vars(args)['verbose'] is not None):
		verbose = vars(args)['verbose']
	else:
		verbose = 2
	print 'verbose: ', verbose
	
	discount_factor_list = [0.1,0.5,0.7,0.9]
	learning_rate_list = [0.1,0.2,0.3,0.4,0.5]
	epsilon_list = [0.0,0.1, 0.3,0.5,0.9,1.0]
	discount_factor=0.9
	learning_rate = 0.5
	if(learning_type == "OFFMC"):
		epsilon = 0.0
	else:
		epsilon = 0.1
	all_averages = []
	average_list = run_episodes([grid_size,grid_size], N, learning_rate, discount_factor, epsilon, softmax=softmax, verbose=verbose, learning_type=learning_type)
	plt.plot(average_list)
	plt.title("Plot of ONMC, eps=0.1, disc=0.9, alpha=0.5")
	plt.ylabel('Rounds needed before catch')
	plt.xlabel('Number of rounds')
	plt.show()

