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

class Game:
	def __init__(self, reset=False, prey=None, predator=None, prey_location=[5,5], predator_location=[0,0], verbose=2, size=[11,11]):
		""" Initalize environment and agents """
		# Initialize environment
		self.environment = Environment(size=size)

		# Initialize prey and predators
		prey_predator_distance = helpers.xy_distance(predator_location, prey_location, self.environment.get_size())
		
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
			print "Round: ", steps
			state = [predator.get_location()[0], predator.get_location()[1], prey.get_location()[0], prey.get_location()[1]]
			caught = self.turn(state)
			self.predator.update_reward(0)
		self.predator.update_reward(10)
		print "Caught prey in " + str(steps) + " rounds!\n=========="
		return steps

	def turn(self, state):
		""" Plays one turn for prey and predator. Choose their action and adjust their state and location accordingly """
		# Play one turn prey
		# Play one turn predator
		prey_location = [state[2], state[3]]
		predator_location = self.turn_predator(state)
		print state
		same = (predator_location == prey_location)
		if(not same):
			prey_location = self.turn_prey()

		#Check if prey is caugh

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
		return new_prey_location

	def turn_predator(self, state):
		""" Perform turn for predator """
		#Remove predator from old location
		self.environment.remove(self.predator.get_location())
		#Get action for predator
		predator_move,action_name = self.predator.action(state)
		#Get new location for predator
		new_predator_location = self.get_new_location(self.predator, predator_move)
		#Move predator to new location
		self.environment.place_object(self.predator, new_predator_location)	
		#Update predator's location in its own knowledge
		self.predator.set_location(new_predator_location)
		return new_predator_location

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

	def value_iteration(self, grid_size, epsilon, discount_factor):
		x_length = grid_size[0]
		y_length = grid_size[1]
		converged = False
		v_grid = np.zeros((x_length, y_length, x_length, y_length))
		temp_grid = np.zeros((x_length, y_length, x_length, y_length))
		delta_grid = np.zeros((x_length, y_length, x_length, y_length))
		actions = self.predator.get_policy([0,0,0,0]).keys()
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
			#helpers.pretty_print(delta_grid[:][:][5][5], label=['delta'])
			#helpers.pretty_print(temp_grid[:][:][5][5], label = ['temp'])
			temp_grid = v_grid
			v_grid = np.zeros((x_length, y_length, x_length, y_length))
			
			#v_grid = np.zeros((x_length, y_length, x_length, y_length))
			delta = np.amax(delta_grid)
			print delta
			print "LOOP: ", loop, "\n"
			
			if(delta < epsilon* (1-discount_factor)/discount_factor):
				converged = True

		helpers.pretty_print(temp_grid[:][:][2][2], label=['v'])
		optimal_policy = self.policy_from_grid(temp_grid, discount_factor, grid_size, False)
		return optimal_policy

	def policy_from_grid(self, grid, discount_factor, grid_size, encoding):
		example_policy = {'North':0, 'East':0, 'South':0, 'West':0, 'Wait':0}
		x_length = grid_size[0]
		y_length = grid_size[1]
		new_policy_grid = [[[[example_policy for i in range(0, y_length)] for j in range(0, x_length)] for k in range(0, y_length)] for l in range(0, x_length)]
		actions = example_policy.keys()  
		for i in range(0, x_length):
				for j in range(0, y_length):
					for k in range(0, x_length):
						for l in range(0, y_length):
							best_action_value = 0
							best_action = ""
							best_actions = []
							for action in actions:
								q_value = self.q_value([i,j,k,l], action, grid, discount_factor, grid_size)
								if(q_value > best_action_value):
									best_action_value = q_value
									best_action = action
									best_actions.append(action)
							new_policy = self.create_optimal_policy(best_actions)
							new_policy_grid[i][j][k][l] = new_policy
		return new_policy_grid

	def encoded_value_iteration(self, grid_size, epsilon, discount_factor):
		x_length = grid_size[0]
		y_length = grid_size[1]
		v_grid = np.zeros((x_length, y_length, x_length, y_length))
		temp_grid = np.zeros((x_length, y_length, x_length, y_length))
		delta_grid = np.zeros((x_length, y_length, x_length, y_length))
		actions = self.predator.get_policy([0,0,0,0]).keys()
		converged = False
		loop = 0
		while(not converged):
			loop +=1
			distance_list = {}
			for i in range(0, x_length):
				for j in range(0, y_length):
					for k in range(0, x_length):
						for l in range(0, y_length):
							distance = [abs(i-k), abs(j-l)]
							if(str(distance) in distance_list):
								v_grid[i][j][k][l] = distance_list[str(distance)]
							else:
								highest_q = 0
								best_action = ""
								for action in actions:
									q_value = self.q_value([i,j,k,l], action, temp_grid, discount_factor, grid_size)
									if q_value > highest_q:
										highest_q = q_value
								v_grid[i][j][k][l] = highest_q
								distance_list[str(distance)] = highest_q

			delta_grid = abs(np.array(v_grid) - np.array(temp_grid))
			#helpers.pretty_print(delta_grid[:][:][5][5], label=['delta'])
			#helpers.pretty_print(temp_grid[:][:][5][5], label = ['temp'])
			temp_grid = v_grid
			v_grid = np.zeros((x_length, y_length, x_length, y_length))
			
			#v_grid = np.zeros((x_length, y_length, x_length, y_length))
			delta = np.amax(delta_grid)
			print delta
			print "LOOP: ", loop, "\n"
			
			if(delta < epsilon* (1-discount_factor)/discount_factor):
				converged = True

		helpers.pretty_print_latex(temp_grid[:][:][5][5], label=[loop,'v'])	
		optimal_policy = self.policy_from_grid(temp_grid, discount_factor, grid_size, False)
		return optimal_policy

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
					if(new_prey == [k,l]):
						transition_prey = 0.8
					else:
						transition_prey = 0.05
					q_value += transition_prey*transition_pred * discount_factor * value_grid[new_state[0]][new_state[1]][new_prey[0]][new_prey[1]]
		return q_value

	def policy_evaluation(self, grid_size, epsilon, discount_factor):
		x_length = grid_size[0]
		y_length = grid_size[1]
		converged = False
		v_grid = np.zeros((x_length, y_length, x_length, y_length))
		temp_grid = np.zeros((x_length, y_length, x_length, y_length))
		delta_grid = np.zeros((x_length, y_length, x_length, y_length))
		actions = {'North':0, 'East':0, 'South':0, 'West':0, 'Wait':0}.keys()
		loop = 0
		while(not converged):
			loop +=1
			for i in range(0, x_length):
				for j in range(0, y_length):
					for k in range(0, x_length):
						for l in range(0, y_length):
							state_value = 0
							policy = self.predator.get_policy([i,j,k,l])
							for action in actions:
								q_value = self.q_value([i,j,k,l], action, temp_grid, discount_factor, grid_size)
								state_value += policy[action] * q_value
							v_grid[i][j][k][l] = state_value
							#print "q_value: ", highest_q

			delta_grid = abs(np.array(v_grid) - np.array(temp_grid))
			temp_grid = v_grid
			v_grid = np.zeros((x_length, y_length, x_length, y_length))
			
			#v_grid = np.zeros((x_length, y_length, x_length, y_length))
			delta = np.amax(delta_grid)
			print delta
			print "LOOP: ", loop, "\n"
			
			if(delta < epsilon* (1-discount_factor)/discount_factor):
				converged = True
				
		return temp_grid

	def encoded_policy_evaluation(self, grid_size, epsilon, discount_factor):
		x_length = grid_size[0]
		y_length = grid_size[1]
		converged = False
		v_grid = np.zeros((x_length, y_length, x_length, y_length))
		temp_grid = np.zeros((x_length, y_length, x_length, y_length))
		delta_grid = np.zeros((x_length, y_length, x_length, y_length))
		actions = {'North':0, 'East':0, 'South':0, 'West':0, 'Wait':0}.keys()
		loop = 0
		while(not converged):
			loop +=1
			distance_list = {}
			for i in range(0, x_length):
				for j in range(0, y_length):
					for k in range(0, x_length):
						for l in range(0, y_length):
							distance = [abs(i-k), abs(j-l)]
							if(str(distance) in distance_list):
								v_grid[i][j][k][l] = distance_list[str(distance)]
							else:
								state_value = 0
								policy = self.predator.get_policy([i,j,k,l])
								for action in actions:
									q_value = self.q_value([i,j,k,l], action, temp_grid, discount_factor, grid_size)
									state_value += policy[action] * q_value
								v_grid[i][j][k][l] = state_value
								distance_list[str(distance)] = state_value
							#print "q_value: ", highest_q

			delta_grid = abs(np.array(v_grid) - np.array(temp_grid))
			temp_grid = v_grid
			v_grid = np.zeros((x_length, y_length, x_length, y_length))
			
			#v_grid = np.zeros((x_length, y_length, x_length, y_length))
			delta = np.amax(delta_grid)
			print delta
			print "LOOP: ", loop, "\n"
			
			if(delta < epsilon* (1-discount_factor)/discount_factor):
				converged = True
		print 0, 0, 5, 5
		print temp_grid[0][0][5][5]
		print 2, 3, 5, 4
		print temp_grid[2][3][5][4]
		print 2, 10, 10, 0
		print temp_grid[2][10][10][0]
		print 10,10,0,0
		print temp_grid[10][10][0][0]
		
		print "loops: ", loop
		return temp_grid

	def policy_iteration(self, grid_size, epsilon, discount_factor):
		evaluated_grid = self.policy_evaluation(grid_size, epsilon, discount_factor)
		policy_grid = predator.get_policy_grid()
		stable = False
		loop = 0
		while(not stable):
			loop +=1
			new_policy_grid, stable = self.policy_improvement(evaluated_grid, policy_grid, grid_size)
			predator.set_policy_grid(new_policy_grid)
			print "In loop ", loop
			if loop > 20:
				break
		return new_policy_grid

	def encoded_policy_iteration(self, grid_size, epsilon, discount_factor):
		evaluated_grid = self.encoded_policy_evaluation(grid_size, epsilon, discount_factor)
		policy_grid = predator.get_policy_grid()
		stable = False
		loop = 0
		while(not stable):
			loop +=1
			new_policy_grid, stable = self.encoded_policy_improvement(evaluated_grid, policy_grid, grid_size)
			evaluated_grid = self.encoded_policy_evaluation(grid_size, epsilon, discount_factor)
			predator.set_policy_grid(new_policy_grid)
			print "In loop ", loop
			if loop > 20:
				break
		helpers.pretty_print_latex(evaluated_grid[:][:][5][5], label=[loop,'v'])	
		return new_policy_grid	

	def policy_improvement(self, v_grid, policy_grid, grid_size):
		x_length = grid_size[0]
		y_length = grid_size[1]
		example_policy = {'North':0, 'East':0, 'South':0, 'West':0, 'Wait':0}
		new_policy_grid = [[[[example_policy for i in range(0, y_length)] for j in range(0, x_length)] for k in range(0, y_length)] for l in range(0, x_length)]
		stability = True
		for i in range(0, x_length):
			for j in range(0, y_length):
				for k in range(0, x_length):
					for l in range(0, y_length):
						backup_policy = policy_grid[i][j][k][l]
						actions = backup_policy.keys()
						best_q_value = 0
						best_action = ""
						best_actions = []
						for action in actions:
							q_value = self.q_value([i,j,k,l], action, v_grid, discount_factor, grid_size)
							if q_value > best_q_value:
								best_q_value = q_value
								best_action = action
								best_actions.append(action)
						new_policy = self.create_optimal_policy(best_actions)
						if(new_policy != backup_policy):
							stability = False
						new_policy_grid[i][j][k][l] = new_policy
		return new_policy_grid, stability

	def encoded_policy_improvement(self, v_grid, policy_grid, grid_size):
		x_length = grid_size[0]
		y_length = grid_size[1]
		example_policy = {'North':0, 'East':0, 'South':0, 'West':0, 'Wait':0}
		new_policy_grid = [[[[example_policy for i in range(0, y_length)] for j in range(0, x_length)] for k in range(0, y_length)] for l in range(0, x_length)]
		stability = True
		distance_list = {}
		for i in range(0, x_length):
			for j in range(0, y_length):
				for k in range(0, x_length):
					for l in range(0, y_length):
						distance = [abs(i-k), abs(j-l)]
						if(str(distance) in distance_list):
							new_policy_grid[i][j][k][l] = distance_list[str(distance)]
						else:
							backup_policy = policy_grid[i][j][k][l]
							actions = backup_policy.keys()
							best_q_value = 0
							best_action = ""
							best_actions = []
							for action in actions:
								q_value = self.q_value([i,j,k,l], action, v_grid, discount_factor, grid_size)
								if q_value > best_q_value:
									best_q_value = q_value
									best_action = action
									best_actions.append(action)
							new_policy = self.create_optimal_policy(best_actions)
							if(new_policy != backup_policy):
								stability = False
							new_policy_grid[i][j][k][l] = new_policy
							distance_list[str(distance)] = new_policy
		return new_policy_grid, stability


	def create_optimal_policy(self, best_actions):
		policy = {'North':0, 'East':0, 'South':0, 'West':0, 'Wait':0}
		for action in best_actions:
			policy[action] = 1.0/len(best_actions)
		return policy


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

#	run_test(alg_type)
	count = 0
	count_list = []
	#Initialize re-usable prey and predator objects
	prey = Prey([2,2])
	predator = Predator([0,0], [2,2])
	
	game = Game(reset=True, prey=prey, predator=predator, verbose=verbose)
	grid_size = [11,11]

	#run_test(prey, predator, game, grid_size, N, discount_factor)
	start_time = time.time()
	#optimal_policy = game.encoded_value_iteration(grid_size, 0.001, 0.1)

	#optimal_policy = game.encoded_policy_iteration(grid_size, 0.001, 0.8)
	#optimal_policy = game.policy_iteration(grid_size, 0.001, 0.8)
	#optimal_policy = game.value_iteration(grid_size, 0.001, 0.8)
	game.encoded_policy_evaluation(grid_size, 0.01, 0.8)
	end_time = time.time()
	predator = Predator([0,0], [2,2], policy=optimal_policy)

'''
	game = Game(reset=True, prey=prey, predator=predator, verbose=0)
	for x in range(0, N):
		# Start game and put prey and predator at initial starting position
		game = Game(reset=True, prey=prey, predator=predator, verbose=verbose, size=grid_size, prey_location=[2,2])
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
	old_result= "Average amount of time steps needed before catch over " + str(N) + " rounds is " + str(average) + ", standard deviation is " + str(standard_deviation)
	print old_result
	print end_time - start_time'''
