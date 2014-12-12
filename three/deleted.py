# deleted stuff in case needed for something:



############################################## From helpers: ##############################################################


#SHOULD BE OBSOLETE
def distance_to_action_ORIGINAL(state, agent_name, new_distance, grid_size=[5,5]):
	"""
	Given the new chosen distance, return corresponding action
	"""
	#predator_state = [current_state[0],current_state[1]]
	#prey_state = [current_state[2],current_state[3]]
	#grid_size = [11,11]
	actions = {'North': [-1,0], 'East': [0,1], 'South': [1,0], 'West': [0,-1], 'Wait': [0,0]}

	dist_to_action = {}
	#need locations dict:
	#Create dictionary of allowed actions (deepcopy actions)
	#For every agent !this agent: compute current distance, compute distance after action
	#If correct, keep action in allowed. if incorrect, remove from allowed
	#If all goes well, at the end there is only one action left!

	# Loop through all possible actions to check for xy_distance

	for name, action in actions.iteritems():
		#Get the result of doing this action, location-wise
		new_state = get_new_location(state[agent_name],action, grid_size=grid_size)
		agent_list = state.keys()
		agent_list.sort()
		amount_correct_actions = 0
		amount_distances = len(new_distance)
		for agent in agent_list:
			if agent != agent_name:
				action_distance = tuple(xy_distance([new_state[0], new_state[1], state[agent][0], state[agent][1]], grid_size))
				if new_distance[int(agent)-1] == action_distance:
					amount_correct_actions+=1
					if amount_correct_actions == amount_distances:
						return name,action
				else:
					break
	random_action = random.choice(actions.keys())
	return random_action, actions[random_action]

#SHOULD BE OBSOLETE
def distance_to_action(state, agent_name, new_distance, grid_size=[5,5]):
	"""
	Given the new chosen distance, return corresponding action
	"""
	#predator_state = [current_state[0],current_state[1]]
	#prey_state = [current_state[2],current_state[3]]
	#grid_size = [11,11]
	actions = {'North': [-1,0], 'East': [0,1], 'South': [1,0], 'West': [0,-1], 'Wait': [0,0]}

	dist_to_action = {}
	#need locations dict:
	#Create dictionary of allowed actions (deepcopy actions)
	#For every agent !this agent: compute current distance, compute distance after action
	#If correct, keep action in allowed. if incorrect, remove from allowed
	#If all goes well, at the end there is only one action left!

	# Loop through all possible actions to check for xy_distance

	possible_actions = []
	for name, action in actions.iteritems():
		#Get the result of doing this action, location-wise
		new_state = get_new_location(state[agent_name],action, grid_size=grid_size)
		agent_list = state.keys()
		agent_list.sort()
		amount_correct_actions = 0
		amount_distances = len(new_distance)
		for agent in agent_list:
			if agent != agent_name:
				action_distance = tuple(xy_distance([new_state[0], new_state[1], state[agent][0], state[agent][1]], grid_size))
				
				# check if we are in the list
				if action_distance in new_distance:
					amount_correct_actions+=1
					# if we are, store action!
					possible_actions.append(name)
					if amount_correct_actions == amount_distances:
						print "name: ", name, " action: ", action
						return name,action
				else:
					continue

	print "PROBLEM! This should never happen!"
	#get second best action
	if len(possible_actions) > 0:
		some_action = random.choice(possible_actions)
		return some_action, actions[some_action]
	
	print "oops! something went wrong! Choosing a random action..."
	random_action = random.choice(actions.keys())
	return random_action, actions[random_action]
	
	
	
	
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




############################################## From newstate: ##############################################################


def relative_xy(self, location1, location2):
        """ Get relative(shortest) distance between two locations using the toroidal property"""
        # Get grid size of the game
        grid_size = self.environment.get_size()
        # Get relative distance to prey using toroidal property
        distance_x = min(abs(state_prey[0] - state_predator[0]), abs(grid_size[0] - abs(state_prey[0] - state_predator[0])))
        distance_y = min(abs(state_prey[1] - state_predator[1]), abs(grid_size[1] - abs(state_prey[1] - state_predator[1])))
        return [distance_x, distance_y]




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

