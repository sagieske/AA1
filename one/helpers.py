import numpy as np
import math

def xy_distance(predator_location, prey_location, grid_size, toroidal=True):
	""" Calculate xy distance using a toroidal grid"""
	x_distance = abs(predator_location[0] - prey_location[0])
	y_distance = abs(predator_location[1] - prey_location[1])

	if toroidal:
		# Make use of toroidal grid
		x_distance = min(x_distance, grid_size[0] - x_distance)
		y_distance = min(y_distance, grid_size[1] - y_distance)
	return [x_distance, y_distance]

def get_rotation(predator_location, prey_location):
	""" Returns unit vector of distance to predator location using prey_location as center """
	x_rotation = predator_location[0] - prey_location[0] 
	y_rotation = predator_location[1] - prey_location[1] 
	return [cmp(x_rotation,0), cmp(y_rotation,0)]

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

