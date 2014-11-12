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



	def full_grid_from_encoding(self, goal_state, encoded_grid, gridsize=[11,11]):
		""" Create full grid from partial grid created by encoded state space"""
		full_grid = np.zeros((gridsize[0],gridsize[1]))

		# Fill in full grid
		for x in range(0,gridsize[0]):
			for y in range(0,gridsize[1]):
				# Get relative distance
				distance = helpers.xy_distance([x,y], goal_state, gridsize)
				# Get value from encoded grid using relative distance
				full_grid[x,y] = encoded_grid[distance[0], distance[1]]
		return full_grid


	def value_encoded(self, discount_factor, start_location_prey=[5,5], gridsize=[11,11], verbose=0):
		""" Use smaller state-space encoding in order to only save 1/3 """
		x_size = gridsize[0]
		y_size = gridsize[1]
		dist_end = helpers.euclidian(start_location_prey, gridsize)
		dist_begin = helpers.euclidian(start_location_prey, [0,0])
		new_grid = np.zeros((x_size,y_size))
		largest_x = 0
		largest_y = 0

		#Find largest difference in x-axis and y-axis from start location of prey to border
		largest_x = max( abs(start_location_prey[0] - 0), abs(start_location_prey[0] - (gridsize[0]-1)))
		largest_y = max( abs(start_location_prey[1] - 0), abs(start_location_prey[1] - (gridsize[1]-1)))

		#The prey's location is at 0,0, because in the encoded grid, each tile is a distance
		#so, distance 0,0 is the prey's location
		new_prey_location = [0,0]
		#Calculate value iteration (mostly) as usual
		self.value_iteration(discount_factor, start_location_prey=new_prey_location, gridsize=[largest_x+1, largest_y+1], encoding=True, verbose=verbose, true_goal_state=start_location_prey, true_gridsize=gridsize)

	def wrap_state(self, state, gridsize, encoding):
		""" Wrap states for non-encoding for toroidal grid"""
		# Only wrap for non-encoding
		if not encoding:
			temp_state = state
			state[0] = temp_state[0] % gridsize[0]
			state[1] = temp_state[1] % gridsize[1]
		return state


	def value_iteration(self,discount_factor, start_location_prey=[5,5], gridsize=[11,11], encoding=False, verbose=0, epsilon=0.000001, true_goal_state=[5,5], true_gridsize=[11,11]):
		""" Calculates value iteration. First gets value grid at convergence, then calculates max policy"""

		# Get value grid
		value_grid = self.get_value_grid(discount_factor, start_location_prey=start_location_prey, gridsize=gridsize, encoding=encoding, verbose=verbose, epsilon=epsilon, true_goal_state=[5,5])

		# Set x and y to true grid size (encoded gridsize can be smaller)
		x_size = true_gridsize[0]
		y_size = true_gridsize[1]

		# TODO: calculate only partial policy grid and use get_rotation to flip the grid?
		# Get all actions
		actions =  self.predator.get_policy().keys()
		# Initialize old policy
		old_policy = {"North":0, "West":0, "East":0, "South":0, "Wait":0}
		policy_grid = [[old_policy for k in range(0, y_size)] for l in range(0, x_size)]

		# Calculate best policy for every position in grid
		for i in range(0, x_size):
			for j in range(0, y_size):
				# Get new states
				possible_new_states = [[i,j], [i+1,j], [i-1,j], [i,j+1], [i,j-1]]
				# Initialize values
				best_action = ""
				best_value = 0
				old_policy = {"North":0, "West":0, "East":0, "South":0, "Wait":0}
				# Calculate values for all possible actions from state. Update best value when value for action is higher
				for action in actions:
					action_value = 0
					# Calculate combined value from all possible new states for current state
					for new_state in possible_new_states:
						# check for wrapping
						new_state = self.wrap_state(new_state, [x_size, y_size], False)
						# Get transition value and reward to new state from current state
						transition_value = self.transition([i,j], new_state, true_goal_state, action)
						reward_value = self.reward_function([i,j], new_state, true_goal_state)
						# Get value grid value from new state
						next_value = value_grid[new_state[0]][new_state[1]]
						# Increase action value with values from new state
						action_value += transition_value * (reward_value + discount_factor * next_value)
					# Set as best value if action value is higher than best encountered value for this action
					if action_value > best_value:
						best_value = action_value
						best_action = action
				# Only set best action to probability of 1 (rest is zero)
				old_policy[best_action] = 1
				# Update policy grid
				policy_grid[i][j] = old_policy

		# print grid
		self.print_policy_grid(policy_grid)
		# needed(for now) since needs to return policy grid
		return value_grid, policy_grid



	def get_value_grid(self, discount_factor, start_location_prey=[5,5], gridsize=[11,11], encoding=False, verbose=0, epsilon=0.000001, true_goal_state=[5,5]):
		""" Calculates value grid until convergence. Returns full grid """
		# Get start time
		start_time = time.time()

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
					if(encoding):
						# new states fall from grid, do not compute
						if new_state[0] == -1 or new_state[1] == -1 or new_state[0] == gridsize[0] or new_state[1] == gridsize[1]:
							continue
					#Check for toroidal wrap
					new_state = self.wrap_state(new_state, [x_size, y_size], encoding)
					# Get value for state (dependent on encoding)
					value = self.get_value(new_state, start_location_prey, discount_factor, [x_size, y_size], value_grid,encoding)
					#print "update grid on %s: %.5f -> %.5f" %(str(new_state), value_grid[new_state[0]][new_state[1]], value)
					# Update grid
					new_grid[new_state[0]][new_state[1]] = value


			# Get delta between old and new grid
			delta_grid = abs(np.array(new_grid) - np.array(value_grid))

			# Update grids for next round
			value_grid = new_grid
			new_grid = np.zeros((x_size,y_size))

			# Get maximum difference between grids
			delta = np.amax(delta_grid)


			count+=1
			# Pretty print dependent on verbose level
			if verbose == 2 or (verbose == 1 and delta < epsilon):
				helpers.pretty_print(value_grid, label=[count, 'Value Iteration Grid '])
			# Check for convergence
			if delta < epsilon:
				convergence = True
				stop_time = time.time()
				print "Value iteration converged! \n- # of iterations: %i\n- Time until convergence in seconds: %.6f" %(count, stop_time-start_time)
				print "Predator location: ", self.predator.get_location()
				print "Prey location: ", start_location_prey
				print "Discount factor: ", discount_factor

		if encoding:
			value_grid = self.full_grid_from_encoding(true_goal_state, value_grid)
			helpers.pretty_print(value_grid, label=['encoded full grid'])
		return value_grid


	def print_policy_grid(self, policy_grid):
		""" Print policy grid using symbols """
		symbols =  {'North': '^', 'East': '>', 'South': 'v','West': '<', 'Wait': '0'}		
		for row in policy_grid:
			# Create string per row
			row_string = []
			for item in row:
				item_string = ''
				# Get movement and translate to symbol
				for key, value in item.iteritems():
					if value != 0:
						item_string += symbols[key]
				row_string.append(item_string)
			# pretty borders
			row_string = '|  ' + '  |  '.join(row_string) +  '  |' 
			print row_string

	
	def next_to_goal(self, state, goal_state):
		""" Function checks if state is next to goal state"""
		x_distance = abs(state[0]- goal_state[0])
		y_distance = abs(state[1]- goal_state[1])
		if x_distance + y_distance ==1:
			return True
		else:
			return False
        		

   	def get_value(self, state, goal_state, discount_factor, grid_size, value_grid, encoding=False):
		""" Get value of a state by using surrounding states and their reward and transition function combined with the discount factor """
		#If the state is the goal_state, the value is 0 because it is terminal
   		if(state == goal_state):
   			return 0
   		#If the state is next to the goal_state, the value is 10 because that is the highest expected reward,
   		#and no matter the discount factor, the value of the corresponding next state is 0.
   		elif (self.next_to_goal(state, goal_state)):
   			return 10
   		else:
   			i = state[0]
	   		j = state[1]
	   		[x_size, y_size] = grid_size
			# Get all actions of predator
	   		actions =  self.predator.get_policy().iteritems()
			action_values = []
			actions_chosen = []
			new_states = [[i,j], [i+1,j], [i-1,j], [i,j+1], [i,j-1]]

			for action in actions:
				prob_sum = 0

				for new_state in new_states:
					# in encoding the x or y distance to the prey cant be smaller than 0 or larger than the gridsize
					if(encoding):
						# Mirror states
						if new_state[0] == -1:
							new_state[0] = 1
						if new_state[1] == -1:
							new_state[1] = 1

						# If at border right or below, then skip
						if new_state[0] == grid_size[0] or new_state[1] == grid_size[1]:
							continue

					#Check for toroidal wrap
					new_state = self.wrap_state(new_state, [x_size, y_size], encoding)

					#Compute transition value from s to s' if not already set
					transition_value = self.transition(state, new_state, goal_state, action[0])

					#Compute reward from s to s'
					reward_value = self.reward_function(state, new_state, goal_state)

					#Add this to the sum of state probabilities
					prob_sum += transition_value * (reward_value + discount_factor * value_grid[new_state[0]][new_state[1]])

				#Append sum of state probabilities for this action times probability for this action to the action list]
				action_values.append(prob_sum*action[1])
				actions_chosen.append(action)

			#The value for i,j is the max of all action_values
			value = max(action_values)

			return value


	def iterative_policy_evaluation(self, discount_factor, start_location_prey=[2,3], gridsize=[11,11], encoding=False, verbose=0):
		""" Performs policy evaluation """
		# Get start time
		start_time = time.time()

		#Initialize parameters
		x_size = gridsize[0]
		y_size = gridsize[1]
		convergence = False

		# Initialize grids
		value_grid = np.zeros((x_size, y_size))
		new_grid = np.zeros((x_size, y_size))
		delta_grid = np.zeros((x_size,y_size))
                
                # Get the predator policy for later use
                policy = self.predator.get_policy()
                         
		count = 0
		# Perform iterative policy evaluation until convergence
		while(not convergence):
			# Calculate the new value for each state of the grid
			
			for i in range(0, x_size):
			    for j in range(0, y_size):
				# current state:
				current_state = [i, j]

				# Get possible actions
				actions =  self.predator.get_policy().keys()
                        
                                # Calculate value
                                value = 0; 
				for action in actions:
				    probability_value = self.get_policy_value(current_state, start_location_prey, discount_factor, [x_size, y_size], value_grid, action, True, encoding)
				    value = value + policy[action] * probability_value
				
				# Update grid
				new_grid[current_state[0]][current_state[1]] = value

			# Get delta between old and new grid
			delta_grid = abs(np.array(new_grid) - np.array(value_grid))

			# Update grids for next round
			value_grid = new_grid
			new_grid = np.zeros((x_size,y_size))

			# Get maximum difference between grids
			delta = np.amax(delta_grid)

			# Pretty print dependent on verbose level
			if verbose == 2 or (verbose == 1 and delta < 0.0001):
				helpers.pretty_print(value_grid, label=[count, 'Value grid '])

			count+=1
			# Check for convergence
			if delta < 0.0001:
				convergence = True
				stop_time = time.time()
				print "Iterative policy evaluation converged! \n- # of iterations: %i\n- Time until convergence in seconds: %.6f" %(count, stop_time-start_time)
				print "Predator location: ", self.predator.get_location()
				print "Prey location: ", start_location_prey
				print "Discount factor: ", discount_factor
		return value_grid


	def policy_evaluation(self, value_grid, policy, start_location_prey, gridsize=[11,11], encoding=False):
		"""
		Perform policy evaluation
		Note: There is a difference between iterative policy evaluation and policy evaluation!
		Use this function to perform policy iteration!
		"""
		convergence = False
		x_size = gridsize[0]
		y_size = gridsize[1]
		new_grid = np.zeros((x_size, y_size))
		delta_grid = np.zeros((x_size,y_size))
            
		# But first perform policy evaluation until convergence!
		# There is a small difference, compared to iterative policy evaluation, hence another implementation           
		while(not convergence):
		# Calculate the new value for each state of the grid
			for i in range(0, x_size):
				for j in range(0, y_size):
					# current state:
					current_state = [i, j]

					# Compute the value by passing the corresponding policy
					value = self.get_policy_value(current_state, start_location_prey, discount_factor, [x_size, y_size], value_grid, policy[i][j], False, encoding)
				
				        # Update grid
				        new_grid[current_state[0]][current_state[1]] = value

			# Get delta between old and new grid
			delta_grid = abs(np.array(new_grid) - np.array(value_grid))
			# Update grids for next round
			value_grid = new_grid
			new_grid = np.zeros((x_size,y_size))
			# Get maximum difference between grids
			delta = np.amax(delta_grid)
			# Check for convergence
			if delta < 0.0001:
				convergence = True
				print helpers.pretty_print(value_grid, label=[count, 'Value grid '])
				return value_grid, delta
		      
	def policy_improvement(self, discount_factor, value_grid, policy, start_location_prey, gridsize=[11,11], encoding=False):
                """ Performs policy improvement """
                
                # Get the optimal policies in a matrix
		updated_policy_matrix = self.get_optimal_policy_matrix(discount_factor, value_grid, start_location_prey, gridsize=[11,11], encoding=False)
                
                # Check if we have reached convergence
		is_policy_stable = updated_policy_matrix == policy
            
                # Return whether the policy is stable and updated policy matrix
		return is_policy_stable, updated_policy_matrix

        def policy_iteration(self, discount_factor, start_location_prey=[2,2], gridsize=[11,11], encoding=False, verbose=0):
		""" Performs policy evaluation """
		# Get start time
		start_time = time.time()

		#Initialize parameters
		x_size = gridsize[0]
		y_size = gridsize[1]

		# Initialize grids
		value_grid = np.zeros((x_size, y_size))

                # Get the predator policy for further use
                old_policy = self.predator.get_policy()
                
                # Initialize policies on grid a "policy grid"
                policy = [[old_policy for i in range(0, y_size)] for j in range(0, x_size)]

                #Initialize variables to keep track of where we are
		count = 0
                is_policy_stable = False
                
                # Perform policy iteration until convergence				
		while(not is_policy_stable):
		      count += 1
		      
		      # Perform policy evaluation
		      value_grid, delta = self.policy_evaluation(value_grid, policy, start_location_prey, gridsize, encoding)
		          
		      # Pretty print value grid, dependent on verbose level
		      if verbose == 2 or (verbose == 1 and delta < 0.0001):
			  	helpers.pretty_print(value_grid, label=[count, 'Value grid '])
		      
		      # Perform policy improvement
		      is_policy_stable, policy = self.policy_improvement(discount_factor, value_grid, policy, start_location_prey, gridsize, encoding)
		      
                      # If policy is not stable, reset whatever necessary for the next round of policy iteration
		      if not is_policy_stable:
		          value_grid = np.zeros((x_size, y_size))
                                
                # print extra information, depending on verbose level
		if verbose == 2 or (verbose == 1 and delta < 0.0001):
		      self.policy_print(policy, value_grid)

                # Stop tracking time! 
                # Print information about this function
		stop_time = time.time()
	        print "Policy iteration converged! \n- # of iterations: %i\n- Time until convergence in seconds: %.6f" %(count, stop_time-start_time)
		print "Predator location: ", self.predator.get_location()
		print "Prey location: ", start_location_prey
		print "Discount factor: ", discount_factor
		
		# Yay! We are done! Return optimal value grid and policy!
		return value_grid, policy

        def get_policy_value(self, state, goal_state, discount_factor, grid_size, value_grid, action_or_policy, policy_evaluation, encoding=False):
		""" Get value of a state by using surrounding states and their reward and transition function combined with the discount factor """
                """ Used for policy evaluation function """
                i = state[0]
                j = state[1]
  	   	[x_size, y_size] = grid_size
 		
 		# TODO: Find a prettier way to pass both action for policy evaluation and policy for policy iteration!!
 		# For now: set both action and policy to whatever was passed, so that we can use the same notation:
 		action = action_or_policy
 		policy = action_or_policy
 		
 		# Get all actions of predator
  	   	new_states = [[i,j], [i+1,j], [i-1,j], [i,j+1], [i,j-1]]
 					
 		prob_sum = 0
                for new_state in new_states:
   		       bool_preset_transition = False
   					
   		       # Currently ignoring the encoding!!
   		       # in encoding the x or y distance to the prey cant be smaller than 0 or larger than the gridsize
   					
   		       if(encoding):
                            # Mirror states
                            if new_state[0] == -1:
                                new_state[0] = 1
 			    if new_state[1] == -1:
 			        new_state[1] = 1
                            
                            # If at border right or below, than use state itself as new state
				"""
				Need to preset transitions since state is adjusted for correct calculation and does not correspond to action:
				Transition should be 1 when action is North/East/South/West since it is a movement to other place 
				(off) the grid. However for correct calculation you need value of state itself. (which would look like action Wait)
				Transition should be 0 when action is Wait.
				"""
  		            if new_state[0] == grid_size[0]:
  		                new_state = state
 			    # pre-set transition_value to 1 if action is not equal to wait
     			        if action != 'Wait':
 			            bool_preset_transition = True
    				    transition_value = 1
 			
 			    #continue
 			    if new_state[1] == grid_size[1]:
 				new_state = state
         			# pre-set transition_value to 1 if action is not equal to wait
         			if action != 'Wait':
            			     bool_preset_transition = True
            			     transition_value = 1
            
   		       # Check for toroidal wrap
   		       new_state = self.wrap_state(new_state, [x_size, y_size], encoding)
   		       
   		       # Compute transition value from s to s' if not already set
   		       # Note: when performing iterative policy evaluation or policy iteration makes a difference!
   		       # Get action vector of action if policy evaluation
   		       if not bool_preset_transition:
   		           if policy_evaluation:
   		               transition_value = self.transition(state, new_state, goal_state, action)
   		           else:
   		               action = self.get_action(state, new_state)
   		               transition_value = policy[action]    
   		       
   		       #Compute reward from s to s'
   		       reward_value = self.reward_function(state, new_state, goal_state)
   		       
   		       #Add this to the sum of state probabilities
   		       prob_sum += transition_value * (reward_value + discount_factor * value_grid[new_state[0]][new_state[1]])
    
 		return prob_sum


	# No doubt this can be implemented smarter, but I have no idea how
	def policy_print(self, policy, value_grid):
		""" Function to print policy matrices in terminal """
		# Assume policy and value_grid are of same dimensions
		policy_strings = self.policy_to_string(policy)
		print "|---------- OPTIMAL POLICY ----------|"
		for (row, pol) in zip(value_grid, policy_strings):
			pretty_row = ['' + '%.4f' %v + ' %s' %z + '' for v, z in zip(row, pol)]
			#print z
			#pdb.set_trace()
			for x in pretty_row:
				print '| ', x[:7], x[7:],
			print ' |\n',   

        # No doubt this can be implemented smarter, but I have no idea how
	def policy_to_string(self, policy):
		""" Function to extract policy to characters => N = North, S = South, etc. """
		# Initialize array with empty strings
		policy_strings = [['' for i in range(0, len(policy))] for j in range(0, len(policy[0]))]
		
		# Iterate over policy array and store only the first letters of the optimal policy at said location
		for i in range(0, len(policy)):
			for j in range(0, len(policy[0])):
				for key in policy[i][j]:
					key_value = policy[i][j][key]
					if (key_value > 0):
						if key == "Wait":
						# Print 'H' of 'Hold' instead of 'Wait'
							policy_strings[i][j] = policy_strings[i][j] + 'H'
						else:
							policy_strings[i][j] = policy_strings[i][j] + key[0]
 				#	print 'policy: ', policy[i][j]            
					#print 'policy string: ', policy_strings[i][j]
				#pdb.set_trace()
		return policy_strings

	def transition(self, old_state, new_state, goal_state, action):
		""" Returns transition states """
		# Get location of new state
		new_location = self.get_new_state_location(old_state, action)
		# Check if transition from old state to new state is possible using action
		if new_location == new_state:     
		      return 1
		else:
		      return 0
			    
			

	def reward_function(self, old_state, new_state, goal_state):
		""" Returns reward at transition to goal state """
		if(new_state == goal_state and old_state != goal_state):
			return 10
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
		          #print 'chosen_move: ', chosen_move, ' action: ', action 
		          return action
		          
	def get_optimal_policy(self, optimal_actions):
		""" Update the policy for policy iteration, by only considering the optimal moves """
		
		new_policy = {'North':0, 'East':0, 'South':0, 'West':0, 'Wait':0}
		
		number_of_optimal_actions = float(len(optimal_actions))
		
		for a in optimal_actions:
		    new_policy[a] = 1/number_of_optimal_actions

		return new_policy
	
	  
	def get_optimal_policy_matrix(self, discount_factor, value_grid, start_location_prey, gridsize=[11,11], encoding=False):
	        """ Calculate the optimal policies per state and return the grid """
	        
	        # Initialize grid
		x_size=gridsize[0]
		y_size=gridsize[1]

		# Declare the policy grid to be returned:
		policy_grid = [[{} for k in range(0, y_size)] for l in range(0, x_size)]

		# Compute the new policy for each state:		      		      		      
		for i in range(0, x_size):
			for j in range(0, y_size):
				# Initialize current state
				current_state = [i, j]
				# Variables to save the current max value and the corresponding optimal moves
				current_max = 0
				current_optimal_actions = []
				# Generate the possible actions
				actions = self.predator.get_action_keys()
				# For each action, compute the corresponding 'weighted reward' and check if
				# it is optimal compared to the explored so far actions
				for action in actions:
					# In order to compute the reward, sum the corresponding rewards for each of the new states
					action_value = 0
					new_states = [[i,j], [i+1,j], [i-1,j], [i,j+1], [i,j-1]]
                        
					# For each new state, compute the reward, multiply it by the corresponding transition function
					# and sum with the previous result:
					for new_state in new_states:
						new_state = self.wrap_state(new_state, [x_size, y_size], encoding)
						immediate_reward = self.reward_function(current_state, new_state, start_location_prey)
						overall_reward = immediate_reward + discount_factor * value_grid[new_state[0]][new_state[1]]
						transition_value = self.transition(current_state, new_state, start_location_prey, action) 
						action_value +=  transition_value * overall_reward
                        
					# Round the values in order to avoid errors in the presicion:
					round_action_value = floor(action_value * (10**3)) / float(10**3)
					round_max = floor(current_max * (10**3)) / float(10**3)
					# If a new max value is found, save it as maximal so far and reset the optimal actions       
					if round_action_value > round_max:
						current_max = action_value
						# Maybe empty the list first?
						current_optimal_actions = [action] 
					# If the new value is better, then check if it's equal to the current maximal   
					elif round_action_value == round_max:
						current_optimal_actions.append(action)

                                # Once the optimal actions for the state are found, update the policy         
                                updated_policy = self.get_optimal_policy(current_optimal_actions)
                                policy_grid[i][j] = updated_policy      
                                
                policy_grid[start_location_prey[0]][start_location_prey[1]] = self.get_optimal_policy(['Wait'])
		return policy_grid



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
	#Run N games
	#TODO: ONLY COMMENTED OUT FOR TESTING PURPOSES
	'''
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
	'''
	#Perform value_iteration over the policy
	#value_grid, policy_grid = game.value_iteration(discount_factor, [5,5], verbose=verbose)
	#game.value_encoded(discount_factor, verbose=verbose)


        #game.iterative_policy_evaluation(discount_factor, [0,0], verbose = verbose)

	
	new_value_grid, new_policy = game.policy_iteration(discount_factor, [5,5], verbose = verbose)
