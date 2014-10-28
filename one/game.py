import random
import math
import sys

'''Predator class, with policy'''
class Predator:
	def __init__(self, location=[0,0]):
		""" Initialize Predator with standard policy """
		self.policy = {'North':0.2, 'East':0.2, 'South':0.2, 'West':0.2, 'Wait':0.2}
		self.actions = {'North': [-1,0], 'East': [0,1], 'South': [1,0],'West': [0,-1], 'Wait':[0,0]}
		self.location = location
		self.state = "Predator(" + str(self.location[0]) + "," + str(self.location[1]) + ")"

	def __repr__(self):
		""" Represent Predator as X """
		return ' X '

	def action(self):
		""" Choose an action and turn it into a move """
		chosen_action = self.pick_action()
		chosen_move = self.actions[chosen_action]
		return chosen_move

	def pick_action(self):
		""" Choose an action from the list at random """
		action = random.choice(self.policy.keys())
		return action

	def get_location(self):
		""" Returns location of predator """
		return self.location

	def set_location(self, new_location):
		""" Set location of predator """
		self.location = new_location
		self.set_state(new_location)

	def get_state(self):
		""" Get state of predator """
		return self.state

	def set_state(self, new_location):
		""" Set state of predator """
		self.state = "Predator(" + str(new_location[0]) + "," + str(new_location[1]) + ")"	

'''Prey class, with policy'''
class Prey:
	def __init__(self, location=[5,5]):
		""" Initialize Prey with standard policy """
		self.policy = {'North':0.05, 'East':0.05, 'South':0.05, 'West':0.05, 'Wait':0.8}
		self.actions = {'North': [-1,0], 'East': [0,1], 'South': [1,0],'West': [0,-1], 'Wait':[0,0]}
		self.location = location
		self.state = "Prey(" + str(self.location[0]) + "," + str(self.location[1]) + ")"

	def __repr__(self):
		""" Represent Prey as 0"""
		return ' O '


	def action(self):
		""" Choose an action and turn it into a move """
		chosen_action = self.pick_action()
		chosen_move = self.actions[chosen_action]
		return chosen_move

	# TODO This can probably be done much better
	def pick_action(self):
		""" Choose an action from the list according to the policy.  """
		threshold = random.uniform(0,100)
		if threshold <= 5:
			return 'North'
		elif threshold <= 10:
			return 'East'
		elif threshold <= 15:
			return 'South'
		elif threshold <= 20:
			return 'West'
		else:
			return 'Wait'
		return action

	def get_location(self):
		""" Return location of prey """
		return self.location		

	def set_location(self, new_location):
		""" Set location of prey """
		self.location = new_location
		self.set_state(new_location)
	
	def get_state(self):
		""" Return state of prey """
		return self.state	

	def set_state(self, new_location):
		""" Set state of prey """
		self.state = "Prey(" + str(new_location[0]) + "," + str(new_location[1]) + ")"	

class Game:

	def __init__(self):
		""" Initialize game """
		self.predator = Predator()
		self.prey = Prey()
		self.environment = Environment()
		#Place prey and predator on board
		self.environment.place_object(self.prey, [5,5])
		self.environment.place_object(self.predator, [0,0])
		self.environment.print_grid()
		#Do steps until prey is caught
		self.rounds = self.until_caught()

	def get_rounds(self):
		""" Return rounds played """
		return self.rounds

	def until_caught(self):
		""" Repeat turns until prey is caught. Returns number of steps until game stopped """
		steps = 0
		caught = 0
		while(caught == 0):
			steps +=1
			caught = self.turn()
		print "Caught prey in " + str(steps) + " rounds!"
		return steps

	def turn(self):
		""" Plays one turn for prey and predator. Choose their action and adjust their state and location accordingly """
		#Remove prey from old location
		self.environment.remove(self.prey.get_location())
		#Get action for prey
		prey_move = self.prey.action()
		#Get new location for prey
		new_prey_location = self.get_new_location(self.prey, prey_move)
		#Check if the prey is not stepping on the predator
		if new_prey_location == self.predator.get_location():
			#If it is, make it wait (hide) instead
			new_prey_location = self.prey.get_location()
			"Prey almost stepped on predator! It's hiding in the bushes instead."
		#Move prey to new location
		self.environment.place_object(self.prey, new_prey_location)	
		#Update prey's location in its own knowledge
		self.prey.set_location(new_prey_location)
		#Remove predator from old location
		self.environment.remove(self.predator.get_location())
		#Get action for predator
		predator_move = self.predator.action()
		#Get new location for predator
		new_predator_location = self.get_new_location(self.predator, predator_move)
		#Move predator to new location
		self.environment.place_object(self.predator, new_predator_location)	
		#Update predator's location in its own knowledge
		self.predator.set_location(new_predator_location)
		#Print the grid
		self.environment.print_grid()
		#Check if prey is caught
		same = (self.predator.get_location() == self.prey.get_location())
		#Show prey & predator states
		print "States: "
		print self.predator.get_state()
		print self.prey.get_state()
		return same

	def get_new_location(self, chosen_object, chosen_move):
		""" Returns new location of an object when performs the chosen move """
		new_location = []
		old_location = chosen_object.get_location()
		environment_size = self.environment.get_size()
		# division by modulo makes board toroidal:
		new_location.append((old_location[0] + chosen_move[0]) % environment_size[0])
		new_location.append((old_location[1] + chosen_move[1]) % environment_size[1])
		return new_location

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
	N = 100
	count = 0
	count_list = []
	for x in range(0, 100):
		game = Game().get_rounds()
		count += game
		count_list.append(game)
	average = float(count/N)
	var_list = [(x-average)**2 for x in count_list]
	variance = float(sum(var_list)/len(var_list))
	standard_deviation = math.sqrt(variance)
	print "Average steps over " + str(N) + " rounds is " + str(average) + ", standard deviation is " + str(standard_deviation)
