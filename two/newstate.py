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
		self.environment = Environment(grid_size, predator_location, prey_location)
		
		if(prey==None):
			prey_policy = Policy(grid_size, prey=True)
			self.prey = Prey(prey_policy)
		else:
			self.prey = prey
			if reset:
				self.environment.place_object(prey, prey_location)

		if(predator==None):
			predator_policy = Policy(grid_size, prey=False)
			self.predator = Predator(predator_policy)
		else:
			self.predator = predator
			if reset:
				self.environment.place_object(predator, predator_location)

		self.verbose = verbose

		if self.verbose > 0:
			self.environment.print_grid()

		print "Episode created with grid size ", grid_size, ", predator at location ", predator_location, ", prey at location ", prey_location

	def get_rounds(self):
		""" Return rounds played """
		self.rounds = self.until_caught()
		return self.rounds

	def until_caught(self):
		""" Repeat turns until prey is caught. Returns number of steps until game stopped """
		steps = 0
		caught = 0
		# Continue until predator has caught the prey
		while(caught == 0):
			steps +=1
			#print "Round: ", steps
			state = self.environment.get_state()
			caught = self.turn(state)
			self.predator.update_reward(0)
		# update predator reward with 10 when caught
		self.predator.update_reward(10)
		print "Caught prey in " + str(steps) + " rounds!\n=========="
		return steps

	def turn(self, state):
		""" Plays one turn for prey and predator. Choose their action and adjust their state and location accordingly """
		# Play one turn prey
		# Play one turn predator
		prey_location = [state[2], state[3]]
		predator_location = self.turn_predator(state)
		#print state
		# Prey is caught
		same = (predator_location == prey_location)
		# If prey is not caught, do prey turn
		if(not same):
			prey_location = self.turn_prey(state)
		# Only print grid or show prey & predator states if verbose level is 1 or 2 
			if (self.verbose == 1 and same):
				self.environment.print_grid()
				print 'States: '
				print 'Predator: ', state[0], ',', state[1]
				print 'Prey: ', state[0], ',', state[1]
				# Always print grid at verbose level 2
			elif self.verbose == 2:
				self.environment.print_grid()
				print 'States: '
				print 'Predator: ', state[0], ',', state[1]
				print 'Prey: ', state[0], ',', state[1]

		return same

	def turn_prey(self, state):
		""" Perform turn for prey """
		prey_move, action_name = self.prey.get_action(state)
		new_location = self.get_new_location('prey', prey_move)
		self.environment.move_object('prey', new_location)
		return new_location

	def turn_predator(self, state):
		""" Perform turn for predator """
		predator_move, action_name = self.predator.get_action(state)
		new_location = self.get_new_location('predator', predator_move)
		self.environment.move_object('predator', new_location)
		return new_location

	def get_new_location(self, chosen_object, chosen_move):
		""" Returns new location of an object when performs the chosen move """
		new_location = []
		old_location = self.environment.get_location(chosen_object)
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
		""" Calculate value iteration. Returns optimal policy grid """
		# Initialize values
		x_length = grid_size[0]
		y_length = grid_size[1]
		converged = False
		v_grid = np.zeros((x_length, y_length, x_length, y_length))
		temp_grid = np.zeros((x_length, y_length, x_length, y_length))
		delta_grid = np.zeros((x_length, y_length, x_length, y_length))
		actions = self.predator.get_policy([0,0,0,0]).keys()
		loop = 0
		# Continue until converged
		while(not converged):
			loop +=1
			# loop over possible states of game(predator(i,j) and prey (k,l))
			for i in range(0, x_length):
				for j in range(0, y_length):
					for k in range(0, x_length):
						for l in range(0, y_length):
							highest_q = 0
							best_action = ""
							# For every action calculate q-value
							for action in actions:
								q_value = self.q_value([i,j,k,l], action, temp_grid, discount_factor, grid_size)
								# Update highest q_value
								if q_value > highest_q:
									highest_q = q_value
							# Update grid
							v_grid[i][j][k][l] = highest_q

			# Calculate difference between grids
			delta_grid = abs(np.array(v_grid) - np.array(temp_grid))

			# Reset temp grid to current value grid for next run
			temp_grid = v_grid
			# Reset value grid to zeros to save all new values in
			v_grid = np.zeros((x_length, y_length, x_length, y_length))
			
			# Get highest difference in delta grid
			delta = np.amax(delta_grid)
			
			# Check if converged
			if(delta < epsilon* (1-discount_factor)/discount_factor):
				converged = True
		# Print last value grid for prey state 2,2
		helpers.pretty_print(temp_grid[:][:][2][2], label=['v'])
		# Calculate optimal policy using value grid
		optimal_policy = self.policy_from_grid(temp_grid, discount_factor, grid_size, False)
		return optimal_policy

	def policy_from_grid(self, grid, discount_factor, grid_size, encoding):
		""" Calculate optimal policy using value grid, returns grid with optimal policy per state """
		# Initialize values
		example_policy = {'North':0, 'East':0, 'South':0, 'West':0, 'Wait':0}
		x_length = grid_size[0]
		y_length = grid_size[1]
		new_policy_grid = [[[[example_policy for i in range(0, y_length)] for j in range(0, x_length)] for k in range(0, y_length)] for l in range(0, x_length)]
		actions = example_policy.keys()  

		# Loop over all possible states (predator (i,j), prey (k,l))
		for i in range(0, x_length):
				for j in range(0, y_length):
					for k in range(0, x_length):
						for l in range(0, y_length):
							# initialize values
							best_action_value = 0
							best_action = ""
							best_actions = []
							# Calculate q value for each action
							for action in actions:
								q_value = self.q_value([i,j,k,l], action, grid, discount_factor, grid_size)
								# Update best actions using q value
								if(q_value > best_action_value):
									best_action_value = q_value
									best_action = action
									best_actions.append(action)
							# Create optimal policy using best actions
							new_policy = self.create_optimal_policy(best_actions)
							# Update policy grid
							new_policy_grid[i][j][k][l] = new_policy
		return new_policy_grid

	def encoded_value_iteration(self, grid_size, epsilon, discount_factor):
		""" Encoded value iteration. Return optimal policy"""
		# Initialize values
		x_length = grid_size[0]
		y_length = grid_size[1]
		v_grid = np.zeros((x_length, y_length, x_length, y_length))
		temp_grid = np.zeros((x_length, y_length, x_length, y_length))
		delta_grid = np.zeros((x_length, y_length, x_length, y_length))
		actions = self.predator.get_policy([0,0,0,0]).keys()
		converged = False
		loop = 0
		# Loop until convergence
		while(not converged):
			loop +=1
			distance_list = {}
			# loop over possible states of game(predator(i,j) and prey (k,l))
			for i in range(0, x_length):
				for j in range(0, y_length):
					for k in range(0, x_length):
						for l in range(0, y_length):
							# Calculate relative distance
							distance = [abs(i-k), abs(j-l)]
							# If relative distance is already calculated, update value grid with its value
							if(str(distance) in distance_list):
								v_grid[i][j][k][l] = distance_list[str(distance)]
							# No know value for this relative distance
							else:
								highest_q = 0
								best_action = ""
								# Calculate q value for all actions
								for action in actions:
									q_value = self.q_value([i,j,k,l], action, temp_grid, discount_factor, grid_size)	
									# Update highest q value
									if q_value > highest_q:
										highest_q = q_value
								# Update value grid
								v_grid[i][j][k][l] = highest_q
								distance_list[str(distance)] = highest_q

			# Calculate difference between old and new value grid
			delta_grid = abs(np.array(v_grid) - np.array(temp_grid))
			# Save value grid to use as old value grid in next run
			temp_grid = v_grid
			# Reset value grid to save new value grid in next run
			v_grid = np.zeros((x_length, y_length, x_length, y_length))
			
			# Get highest difference from delta grid
			delta = np.amax(delta_grid)
			
			# Check for convergence
			if(delta < epsilon* (1-discount_factor)/discount_factor):
				converged = True

		# Print grid for prey state 5,5
		helpers.pretty_print(temp_grid[:][:][2][2], label=[loop,'v'])

		# Calculate optimal policy
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

	def policy_evaluation(self, grid_size, epsilon, discount_factor):
		""" Calculate policy evaluation"""
		# initialize values
		x_length = grid_size[0]
		y_length = grid_size[1]
		converged = False
		v_grid = np.zeros((x_length, y_length, x_length, y_length))
		temp_grid = np.zeros((x_length, y_length, x_length, y_length))
		delta_grid = np.zeros((x_length, y_length, x_length, y_length))
		actions = {'North':0, 'East':0, 'South':0, 'West':0, 'Wait':0}.keys()
		loop = 0
		# Loop until converged
		while(not converged):
			loop +=1
			# loop over possible states of game(predator(i,j) and prey (k,l))
			for i in range(0, x_length):
				for j in range(0, y_length):
					for k in range(0, x_length):
						for l in range(0, y_length):
							state_value = 0
							policy = self.predator.get_policy([i,j,k,l])
							# Calculate q value for all actions
							for action in actions:
								q_value = self.q_value([i,j,k,l], action, temp_grid, discount_factor, grid_size)
								# Update value of state using action probability and q value of action
								state_value += policy[action] * q_value
							# Update value grid
							v_grid[i][j][k][l] = state_value

			# Calculate difference between old and new value grid
			delta_grid = abs(np.array(v_grid) - np.array(temp_grid))
			# Save value grid for next run
			temp_grid = v_grid
			# Reset v grid to save new in value grid
			v_grid = np.zeros((x_length, y_length, x_length, y_length))
			
			# Calculate highest difference in delta grid
			delta = np.amax(delta_grid)

			# Check convergence			
			if(delta < epsilon* (1-discount_factor)/discount_factor):
				converged = True

		helpers.pretty_print(temp_grid[:][:][2][2], label=[loop,'v'])
		return temp_grid

	def encoded_policy_evaluation(self, grid_size, epsilon, discount_factor):
		""" Calculate policy evaluation for encoded grid """
		# Initialize values
		x_length = grid_size[0]
		y_length = grid_size[1]
		converged = False
		v_grid = np.zeros((x_length, y_length, x_length, y_length))
		temp_grid = np.zeros((x_length, y_length, x_length, y_length))
		delta_grid = np.zeros((x_length, y_length, x_length, y_length))
		actions = {'North':0, 'East':0, 'South':0, 'West':0, 'Wait':0}.keys()
		loop = 0
		# Loop until converged
		while(not converged):
			loop +=1
			distance_list = {}
			# loop over possible states of game(predator(i,j) and prey (k,l))
			for i in range(0, x_length):
				for j in range(0, y_length):
					for k in range(0, x_length):
						for l in range(0, y_length):
							# Get relative distance
							distance = [abs(i-k), abs(j-l)]
							# If value for relative distance is already known use this to update value grid
							if(str(distance) in distance_list):
								v_grid[i][j][k][l] = distance_list[str(distance)]
							# Value for relative distance is unknown
							else:
								state_value = 0
								policy = self.predator.get_policy([i,j,k,l])
								# Loop over actions to update state value using action probability and qvalue of action in state
								for action in actions:
									q_value = self.q_value([i,j,k,l], action, temp_grid, discount_factor, grid_size)
									state_value += policy[action] * q_value
								# Update value grid
								v_grid[i][j][k][l] = state_value
								# Add to distance_list dict
								distance_list[str(distance)] = state_value

			# Calculate difference
			delta_grid = abs(np.array(v_grid) - np.array(temp_grid))
			# Reset grids
			temp_grid = v_grid
			v_grid = np.zeros((x_length, y_length, x_length, y_length))
			
			# Calculate highest difference in delta grid
			delta = np.amax(delta_grid)

			# Check for convergence 			
			if(delta < epsilon* (1-discount_factor)/discount_factor):
				converged = True
		helpers.pretty_print(temp_grid[:][:][2][2], label=[loop,'v'])
		return temp_grid

	def policy_iteration(self, grid_size, epsilon, discount_factor):
		""" Calculate policy iteration, return improved policy grid"""
		# Get policy evaluated grid
		evaluated_grid = self.policy_evaluation(grid_size, epsilon, discount_factor)
		policy_grid = predator.get_policy_grid()
		stable = False
		loop = 0
		# Continue until stable policy grid
		while(not stable):
			loop +=1
			# Calculate policy grid and check if stable
			new_policy_grid, stable = self.policy_improvement(evaluated_grid, policy_grid, grid_size)
			# Set policy grid for predator
			predator.set_policy_grid(new_policy_grid)
			print "In loop ", loop
			if loop > 20:
				break
		return new_policy_grid

	def encoded_policy_iteration(self, grid_size, epsilon, discount_factor):
		""" Calculate encoded policy iteration, return improved policy grid"""
		# Get policy evaluation grid using encoding
		evaluated_grid = self.encoded_policy_evaluation(grid_size, epsilon, discount_factor)
		policy_grid = predator.get_policy_grid()
		stable = False
		loop = 0
		# Continue until stable policy grid
		while(not stable):
			loop +=1
			# Calculate policy grid and check if stable
			new_policy_grid, stable = self.encoded_policy_improvement(evaluated_grid, policy_grid, grid_size)
			evaluated_grid = self.encoded_policy_evaluation(grid_size, epsilon, discount_factor)
			# Set policy grid for predator
			predator.set_policy_grid(new_policy_grid)
			print "In loop ", loop
			if loop > 20:
				break
		helpers.pretty_print(evaluated_grid[:][:][2][2], label=[loop,'v'])	
		return new_policy_grid	

	def policy_improvement(self, v_grid, policy_grid, grid_size):
		""" Calculate policy improvement, return improved policy grid and stability"""
		# Initialize
		x_length = grid_size[0]
		y_length = grid_size[1]
		example_policy = {'North':0, 'East':0, 'South':0, 'West':0, 'Wait':0}
		new_policy_grid = [[[[example_policy for i in range(0, y_length)] for j in range(0, x_length)] for k in range(0, y_length)] for l in range(0, x_length)]
		stability = True
		# TODO
		# loop over possible states of game(predator(i,j) and prey (k,l))
		for i in range(0, x_length):
			for j in range(0, y_length):
				for k in range(0, x_length):
					for l in range(0, y_length):
						# backup current policy and actions
						backup_policy = policy_grid[i][j][k][l]
						actions = backup_policy.keys()
						best_q_value = 0
						best_action = ""
						best_actions = []
						# Calculate q value for each action
						for action in actions:
							q_value = self.q_value([i,j,k,l], action, v_grid, discount_factor, grid_size)
							# Update best q values
							if q_value > best_q_value:
								best_q_value = q_value
								best_action = action
								best_actions.append(action)
						# Get optimal policy
						new_policy = self.create_optimal_policy(best_actions)
						# If optimal policy is not equal to current policy, it is not stable
						if(new_policy != backup_policy):
							stability = False
						# Update new policy grid
						new_policy_grid[i][j][k][l] = new_policy
		return new_policy_grid, stability

	def encoded_policy_improvement(self, v_grid, policy_grid, grid_size):
		""" Calculate encoded policy improvement, return improved policy grid and stability"""
		# Initalize 
		x_length = grid_size[0]
		y_length = grid_size[1]
		example_policy = {'North':0, 'East':0, 'South':0, 'West':0, 'Wait':0}
		new_policy_grid = [[[[example_policy for i in range(0, y_length)] for j in range(0, x_length)] for k in range(0, y_length)] for l in range(0, x_length)]
		stability = True
		distance_list = {}
		# loop over possible states of game(predator(i,j) and prey (k,l))
		for i in range(0, x_length):
			for j in range(0, y_length):
				for k in range(0, x_length):
					for l in range(0, y_length):
						# Calculate relative distance
						distance = [abs(i-k), abs(j-l)]
						# If value for relative distance is known, use this to update policy grid
						if(str(distance) in distance_list):
							new_policy_grid[i][j][k][l] = distance_list[str(distance)]
						# Unknown value for relative distance
						else:
							# backup current policy and actions
							backup_policy = policy_grid[i][j][k][l]
							actions = backup_policy.keys()
							best_q_value = 0
							best_action = ""
							best_actions = []
							# Calculate q value for each action
							for action in actions:
								q_value = self.q_value([i,j,k,l], action, v_grid, discount_factor, grid_size)
								# Update best q values
								if q_value > best_q_value:
									best_q_value = q_value
									best_action = action
									best_actions.append(action)
							# Get optimal policy
							new_policy = self.create_optimal_policy(best_actions)
							# If optimal policy is not equal to current policy, it is not stable
							if(new_policy != backup_policy):
								stability = False
							# Update policy grid and distance_list
							new_policy_grid[i][j][k][l] = new_policy
							distance_list[str(distance)] = new_policy
		return new_policy_grid, stability


	def create_optimal_policy(self, best_actions):
		""" Calculate optimal policy given actions"""
		policy = {'North':0, 'East':0, 'South':0, 'West':0, 'Wait':0}
		for action in best_actions:
			policy[action] = 1.0/len(best_actions)
		return policy

def run_episode(policy, predator):
		game = Game()
		game.get_rounds()

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

	run_episode("policy", "predator")