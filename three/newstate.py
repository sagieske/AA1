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
import matplotlib.pyplot as plt

class Game:
	def __init__(self, prey=None, predator=None, predator_location=[0,0], prey_location=[5,5], softmax=False, grid_size=[11,11], learning_type='Q-learning', predator_list=None):
		""" Initalize environment and agents """
		print "Learning algorithm used in this episode: ", learning_type
		
		#Store the learning type
		self.learning_type = learning_type
		#Instantiate environment object with correct size, predator and prey locations
		self.environment = Environment(grid_size, predator_location, prey_location)
		#Create prey if none was given
		if(prey==None):
			prey_policy = Policy(grid_size, prey=True, softmax=softmax)
			self.prey = Prey(prey_policy, len(predator_list)+1)
		#Else, store prey
		else:
			self.prey = prey
		#Create predator if none was given
		if(predator_list==None):
			predator_policy = Policy(grid_size, prey=False, softmax=softmax)
			self.predator_list = [Predator(predator_policy)]
		#Else, store the predator
		else:
			self.predator_list = predator_list

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
			newstate = self.environment.get_state()
		
		print "Caught prey in " + str(steps) + " rounds!\n=========="
		distance_dict_test = self.predator.get_policy_grid().distance_dict
		#print self.predator.get_policy_grid().distance_dict
			
		#return steps, self.predator.get_policy_grid()
		return steps,  self.predator.get_policy_grid(), distance_dict_test

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
		#If predator moves into the prey, the prey is caught
		same = (predator_location == prey_location)

		#If we're using q-learning, update the q-values using a greedy action in next state
		if(self.learning_type == 'Q-learning'):
			self.predator.q_learning(predator_action, old_state, new_state, learning_rate, discount_factor, epsilon)
		#If we're using Sarsa, update the q-values using the policy to select an action in the next state
		elif(self.learning_type == 'Sarsa'):
			action = self.predator.sarsa(predator_action, old_state, new_state, learning_rate, discount_factor, epsilon)

		if(not same):
			#If prey is not caught, move it
			prey_location = self.turn_prey(old_state, predator_location, epsilon)
			#Print effect of this turn
			if (same):
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
		
		# The prey can trip 20% of the time
		if action_name is not "Wait":
			if random.randint(1, 10) <= 2:
				prey_move = [0,0]
				action_name = 'Trip'

		#Turn action into new location
		new_location = self.get_new_location('prey', prey_move)
		#Move the prey to the new location
		self.environment.move_object('prey', new_location)
		return new_location

	def turn_predator(self, state):
		""" Perform turn for predator """
		#Retrieve the action for the predator for this state
		#If we're using off-policy mc, use completely random action
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

def run_episodes(grid_size, N, learning_rate, discount_factor, epsilon, amount_predators=2, softmax=False, verbose=0, learning_type='Q-learning'):
	""" Run N episodes and compute average """
	for y in range(0, 100):
		predator_list = []
		for i in range(0, amount_predators):
			pred_pol = Policy(grid_size, amount_agents=amount_predators+1)
			predator_list.append(Predator(pred_pol))
		total_rounds = 0
		rounds_list = []
	#If we're using off-policy MC, initialize game/predator differently to allow separated learn/test runs
		game = Game(grid_size=grid_size, softmax=softmax, learning_type=learning_type, predator_list=predator_list)
		average_list = []
		counter=0
		current_rounds=0
	
		y_list = []
		for x in range(0, N):
			print "Rounds needed to catch prey: ", current_rounds
			#Initialize episode
			#If we're using off-policy MC, initialize a learning and then a testing episode
			#current_rounds, policy_grid = game.get_rounds(learning_rate, discount_factor, epsilon)
			current_rounds, policy_grid, distance_dict = game.get_rounds(learning_rate, discount_factor, epsilon)
			#predator = Predator(policy_grid)
			predator = Predator(policy_grid)
			policy_grid_pred = predator.get_policy_grid()
			policy_grid_pred.set_distance_dict(distance_dict)

			print "item 2,2"
			print policy_grid_pred.distance_dict[(2,2)]
			print "item 1,1"
			print policy_grid_pred.distance_dict[(1,1)]
			print "item 0,1"
			print policy_grid_pred.distance_dict[(0,1)]

			#print policy_grid.get_policy([4,4,5,5])
			#print policy_grid.get_policy([5,5,6,6])
			#If we're using on-policy Monte Carlo, calculate the average using the returns for each state,action pair
			
			print "Finished episode: ", x
			game = Game(grid_size=grid_size, predator=predator, softmax=softmax, learning_type=learning_type, predator_list=predator_list)	

			#Add rounds needed in test episode to total_rounds	
			total_rounds += current_rounds
			#Add rounds needed in this episode to the list of rounds
			rounds_list.append(current_rounds)
		y_list.append(rounds_list)
		#Smooth graph
	average_rounds = []
	for number in range(0, len(rounds_list)):
		yl_number = 0
		for yl in y_list:
			yl_number += yl[number]
		average_rounds.append(yl_number)

	print "List of steps needed per episode: ", rounds_list
	print "List of smoothed averages: ", average_rounds
	#Compute average number of rounds needed
	#average_rounds = float(average_rounds)/N
	#Compute list of variances
	#var_list = [(x-average_rounds)**2 for x in rounds_list]
	#Compute average variance
	#variance = float(sum(var_list)/len(var_list))
	#Compute standard deviation for N rounds
	#standard_deviation = math.sqrt(variance)
	#print "Average rounds needed over ", N, " episodes: ", average_rounds
	#print "Standard deviation: ", standard_deviation	
	return average_rounds


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
	parser.add_argument('-predators', metavar='Specify number of predators', type=int)
	args = parser.parse_args()

	#Default parameter values
	N = 100
	discount_factor = 0.9
	learning_rate = 0.5
	epsilon = 0.1
	grid_size = 11
	softmax = False	
	learning_type = "Q-learning"
	amount_predators = 2

	#Command line parsing
	if(vars(args)['runs'] is not None):
		N = vars(args)['runs']
	if(vars(args)['learning_type'] is not None):
		learning_type = vars(args)['learning_type']
	if(vars(args)['runs'] is not None):
		N = vars(args)['runs']
	if(vars(args)['predators'] is not None):
		amount_predators = vars(args)['predators']
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


	all_averages = []
	average_list = run_episodes([grid_size,grid_size], N, learning_rate, discount_factor, epsilon, amount_predators=amount_predators, softmax=softmax, learning_type=learning_type)
	plt.plot(average_list)
	plt.title("Steps needed versus episode number")
	plt.ylabel('Steps needed before catch')
	plt.xlabel('Number of steps')
	plt.show()

