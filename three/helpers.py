import numpy as np
import math
from operator import add,mul, sub
import sys
import time

def distance_to_action(current_state):
	"""
	Given the new chosen distance, return corresponding action
	"""
	predator_state = [current_state[0],current_state[1]]
	prey_state = [current_state[2],current_state[3]]
	grid_size = [11,11]
	actions = {'North': [-1,0], 'East': [0,1], 'South': [1,0], 'West': [0,-1], 'Wait': [0,0]}

	dist_to_action = {}
	# Loop through all possible actions to check for xy_distance
	for name, action in actions.iteritems():
		# get new state
		new_state = get_new_location(predator_state,action)
		# get xydistance
		action_distance = xy_distance([new_state[0], new_state[1]], prey_state, [11,11])
		dist_to_action[tuple(action_distance)] = (name, new_state)
	return dist_to_action

# ONLY NEEDED FOR DEBUG
def action_to_distance(current_state, action):
	predator_state = [current_state[0],current_state[1]]
	prey_state = [current_state[2],current_state[3]]
	grid_size = [11,11]
	new_state = get_new_location(predator_state,action)
	action_distance = xy_distance([new_state[0], new_state[1]], prey_state, [11,11])



def get_new_location(object_location, transformation):
	""" Returns new location of an object when performs the chosen move """
	new_location = []
	#Retrieve the agent's position in the grid
	#Get the size of the environment
	environment_size = [11,11]
	#Wrap edges to make grid toroidal
	new_location.append((object_location[0] + transformation[0]) % environment_size[0])
	new_location.append((object_location[1] + transformation[1]) % environment_size[1])
	return new_location

	

def timer(start_time, name):
	""" Stops timer and prints total time of function
	PUT THIS IN BEGINNING OF FUNCTION:
	# start timer
	start_time = time.time()

	AT END OF FUNCTION:
	helpers.timer(start_time, <function_name>)
	"""
	end_time = time.time() - start_time
	print name, " - time taken: ", end_time

def xy_distance(state, grid_size, toroidal=True):
	""" Calculate xy distance using a toroidal grid"""
	x_distance = abs(state[0] - state[2])
	y_distance = abs(state[1] - state[3])

	if toroidal:
		# Make use of toroidal grid
		x_distance = min(x_distance, grid_size[0] - x_distance)
		y_distance = min(y_distance, grid_size[1] - y_distance)
	return [x_distance, y_distance]



def get_rotation(predator_location, prey_location):
	""" Returns unit vector of distance to predator location using prey_location as center """
	x_rotation = predator_location[0] - prey_location[0] 
	y_rotation = predator_location[1] - prey_location[1] 
	rot_x = cmp(x_rotation,0)
	rot_y = cmp(y_rotation,0)
	if rot_x == 0:
		rot_x = 1
	if rot_y == 0:
		rot_y = 1
	return [rot_x, rot_y]

def euclidian(first_location, second_location):
	"""  Calculates euclidian distance"""
	distance = math.sqrt((first_location[0]-second_location[0])**2 + (first_location[1]-second_location[1])**2)
	return distance

def pretty_print(matrix, label=None, dec=7):
	""" Function to pretty print matrices in terminal """
	# Get max size of value 
	max_value = '%i' %(np.amax(matrix))
	if label is not None:
		if len(label) == 1:
			print "|----------", label[0], "----------|"
		elif len(label) == 2:
			print "|----------", label[1], " in loop ", label[0], "----------|"
	# Create string to pad float
	f_string = '%' +str(dec+len(max_value)+1)+'.'+str(dec)+'f'
	for row in matrix:
		pretty_row = [f_string %v for v in row]
		print '| ', ' | '.join(pretty_row), ' |'

def policy_print_latex(policy_grid, label, indices=True):
	""" Function to pretty print policies in terminal to copy to laTeX"""
	#also only print one optimal function
	symbols =  {'North': '\\textasciicircum', 'East': '\\textgreater', 'South': 'v','West': '\\textless', 'Wait': 'X'}
	
	if indices:
		tab_array = ['l'] * (len(policy_grid)+1)
	else:
		tab_array = ['l'] * len(policy_grid)

	tab_array = ['l'] * len(policy_grid)
	tab = ' | '.join(tab_array)
	print "\\begin{tabular}{ |"+ tab + "|}"
	# Print title
	print "\\hline"
	multicolumn = len(policy_grid)
	if indices:
		multicolumn_str = str(len(policy_grid)+1)		
	else:
		multicolumn_str = str(len(policy_grid))
	print "\multicolumn{"+ multicolumn_str +"}{|c|}{" + label[1] + " in loop " + str(label[0]) + "}\\\\"
	print "\\hline"
	# Indices x-axis
	index = range(0,len(policy_grid))
	index_str = ["%s" % str(x) for x in index]
	index_str_line =  ' & '.join(index_str) + ' \\\\ \n'
	if indices:
		index_str_line = "Indices y\\textbackslash x &" + index_str_line
	print index_str_line
	
	print "\\hline"
	# Print rows
	index = 0
	for row in policy_grid:
		# Create string per row
		row_string = []
		for item in row:
			item_string = ''
			# Get movement and translate to symbol
			for key, value in item.iteritems():
				if value != 0:
					item_string += symbols[key]
					break;
			row_string.append(item_string)
		# pretty borders
		row_string = ' & '.join(row_string)
		if indices:
			row_string = str(index) + ' & ' + row_string + '\\\\'
		print row_string
		index = index + 1
	print "\\hline"
	print "\\end{tabular}"

def pretty_print_latex(matrix, label, indices=True):
	""" Function to pretty print matrices in terminal to copy to laTeX"""
	print "|----------", label[1], " in iteration ", label[0], "----------|"
	
	# Begin tabular
	if indices:
		tab_array = ['l'] * (len(matrix)+1)
	else:
		tab_array = ['l'] * len(matrix)
	tab = ' | '.join(tab_array)
	print "\\begin{tabular}{ |"+ tab + "|}"
		# Print title
	print "\\hline"
	multicolumn = len(matrix)
	if indices:
		multicolumn_str = str(len(matrix)+1)		
	else:
		multicolumn_str = str(len(matrix))
	print "\multicolumn{"+ multicolumn_str +"}{|c|}{" + label[1] + " in loop " + str(label[0]) + "}\\\\"
	print "\\hline"
	# Indices x-axis
	index = range(0,len(matrix))
	index_str = ["%s" % str(x) for x in index]
	index_str_line =  ' & '.join(index_str) + ' \\\\ \n'
	if indices:
		index_str_line = "Indices y\\textbackslash x &" + index_str_line
	print index_str_line
	
	print "\\hline"
	# Print rows
	for index in range(0,len(matrix)):
		pretty_row = ['%.6f' %v  for v in matrix[index]]
		if indices:
			pretty_row = [str(index)] + pretty_row
		latex_row = ' & '.join(pretty_row)
		latex_row_new = latex_row + ' \\\\'
		print latex_row_new
	# End tabular 
	print "\\hline"
	print "\\end{tabular}"


def full_policy_grid_from_encoding(goal_state, encoded_grid, gridsize=[11,11]):
	""" Create full grid from partial grid created by encoded state space for policy grid"""
	full_grid = np.zeros((gridsize[0],gridsize[1]))

	old_policy = {"North":0, "West":0, "East":0, "South":0, "Wait":0}
	full_grid = [[old_policy for k in range(0, gridsize[1])] for l in range(0,  gridsize[0])]

	# Fill in full grid
	for x in range(0,gridsize[0]):
		for y in range(0,gridsize[1]):
			# Get relative distance
			distance = xy_distance([x,y], goal_state, gridsize)
			# Get true distance
			true_distance = map(sub, [x,y], goal_state)

			# Get rotation
			rotation =  get_rotation([x,y], goal_state)

			# Wanna flip rotation if relative distance is smaller than normal distance (due to toroidality)
			if distance[0] < abs(true_distance[0]):
				rotation[0] = -(rotation[0])

			if distance[1] < abs(true_distance[1]):
				rotation[1] = -(rotation[1])

			# Get policy of state
			policy_state = encoded_grid[distance[0]][distance[1]]
			new_policy = flip_policy(policy_state,rotation)

			# Flip all policies accordingly
			# Get value from encoded grid using relative distance
			full_grid[x][y] = new_policy	
	return full_grid

def flip_policy(policy, rotation):
	"""Rotates actions according to 'angle' from prey location"""
	actions = {'North': [-1,0], 'East': [0,1], 'South': [1,0],'West': [0,-1], 'Wait':[0,0]}
	action_name_list, movement = zip(*actions.items())
	# Create copy for new policy
	new_policy = policy.copy()
	# rotate actions
	for index in range(0,len(movement)):
		old_action = action_name_list[index]
		old_movement = movement[index]
		new_movement = map(mul, rotation, old_movement)
		index_new_movement = movement.index(new_movement)
		new_action = action_name_list[index_new_movement]

		# Flip values
		if old_action != new_action and policy[old_action] == 1:
			new_policy[old_action] = 0
			new_policy[new_action] = 1

	return new_policy
	

def get_optimal_action(policy):
    """ Use the probabilities in the policy to pick the optimal move """
    # Get highest probability
    optimal = max(policy.values())
    
	# Return action using highest probability
    for each in policy.keys():
        if policy[each] == optimal:
            return each

