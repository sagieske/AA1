def pretty_print(matrix, label):
	""" Function to pretty print matrices in terminal """
	print "|----------", label[1], " in loop ", label[0], "----------|"
	for row in matrix:
		pretty_row = ['%.6f' %v +'|' for v in row]
		for x in pretty_row:
			print '| ', x[:7],
		print ' |\n',

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

