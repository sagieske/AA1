class PredatorvsPrey:

	# Global variables for states of predator and prey
	predator = None
	prey = None
	# Global variable for grid size
	gridsize = None

	# Global dictionary for actions
	# TODO? seperate actions prey and predator
	actions_dict = {'North': (-1,0),'West': (0,-1),'South': (1,0),'East': (0,1)}

	def __init__(self, gridsize = (11,11), startPrey = (0,0), startPredator = (5,5)):
		""" Initialize grid and predator and prey starting positions """
		self.predator = startPredator;
		self.prey = startPrey;
		self.gridsize = gridsize;

	def move(self, agent, action):
		""" Move agent according to action """
		pass

	def get_location(self, currentlocation, action):
		""" Returns location corresponding to action on current location. Grid is toroidal"""
		# Retrieve movement tuple from dictionary
		movement = self.actions_dict[action]
		# Get new location using modulo of gridsize
		newlocation = ((currentlocation[0]+movement[0]) % self.gridsize[0], (currentlocation[1]+movement[1]) % self.gridsize[1])

		return newlocation

		
	def print_locations(self):
		 print "Gridsize: %s" %(str(self.gridsize)) 
		 print "Predator is at location: %s" %(str(self.predator)) 
		 print "Prey is at location: %s" %(str(self.prey))

game = PredatorvsPrey()
game.print_locations() 
game.get_location((1,1), 'South')

