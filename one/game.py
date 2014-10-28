import random
import math

'''Predator class, with policy'''
class Predator:
	#Initialize Predator with standard policy
	def __init__(self, location=[0,0]):
		self.policy = {'North':0.2, 'East':0.2, 'South':0.2, 'West':0.2, 'Wait':0.2}
		self.actions = {'North': [-1,0], 'East': [0,1], 'South': [1,0],'West': [0,-1], 'Wait':[0,0]}
		self.location = location
		self.state = "Predator(" + str(self.location[0]) + "," + str(self.location[1]) + ")"

	#Represent Predator as X
	def __repr__(self):
		return ' X '

	#Choose an action and turn it into a move
	def action(self):
		chosen_action = self.pick_action()
		chosen_move = self.actions[chosen_action]
		return chosen_move

	#Choose an action from the list at random
	def pick_action(self):
		action = random.choice(self.policy.keys())
		return action

	def get_location(self):
		return self.location

	def set_location(self, new_location):
		self.location = new_location
		self.set_state(new_location)

	def get_state(self):
		return self.state

	def set_state(self, new_location):
		self.state = "Predator(" + str(new_location[0]) + "," + str(new_location[1]) + ")"	

'''Prey class, with policy'''
class Prey:
	#Initialize Prey with standard policy
	def __init__(self, location=[5,5]):
		self.policy = {'North':0.05, 'East':0.05, 'South':0.05, 'West':0.05, 'Wait':0.8}
		self.actions = {'North': [-1,0], 'East': [0,1], 'South': [1,0],'West': [0,-1], 'Wait':[0,0]}
		self.location = location
		self.state = "Prey(" + str(self.location[0]) + "," + str(self.location[1]) + ")"

	#Represent Prey as 0
	def __repr__(self):
		return ' O '
	#Choose an action and turn it into a move
	def action(self):
		chosen_action = self.pick_action()
		chosen_move = self.actions[chosen_action]
		return chosen_move

	#Choose an action from the list according to the policy
	#This can probably be done much better
	def pick_action(self):
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
		return self.location		

	def set_location(self, new_location):
		self.location = new_location
		self.set_state(new_location)
	
	def get_state(self):
		return self.state	

	def set_state(self, new_location):
		self.state = "Prey(" + str(new_location[0]) + "," + str(new_location[1]) + ")"	

class Game:
	#Initialize game 
	def __init__(self):
		self.predator = Predator()
		self.prey = Prey()
		self.environment = Environment()
		self.environment.place_object(self.prey, [5,5])
		self.environment.place_object(self.predator, [0,0])
		self.environment.print_grid()
		self.rounds = self.until_caught()

	def get_rounds(self):
		return self.rounds

	def until_caught(self):
		steps = 0
		caught = 0
		while(caught == 0):
			steps +=1
			caught = self.turn()
		print "Caught prey in " + str(steps) + " rounds!"
		return steps

	def turn(self):
		self.environment.remove(self.prey.get_location())
		prey_move = self.prey.action()

		new_prey_location = self.move(self.prey, prey_move)
		if new_prey_location == self.predator.get_location():
			new_prey_location = self.prey.get_location()
			"Prey almost stepped on predator! It's hiding in the bushes instead."

		self.environment.place_object(self.prey, new_prey_location)	
		self.prey.set_location(new_prey_location)
		self.environment.remove(self.predator.get_location())
		predator_move = self.predator.action()
		new_predator_location = self.move(self.predator, predator_move)
		self.environment.place_object(self.predator, new_predator_location)	
		self.predator.set_location(new_predator_location)
		self.environment.print_grid()
		same = (self.predator.get_location() == self.prey.get_location())
		print "States: "
		print self.predator.get_state()
		print self.prey.get_state()
		return same

	def move(self, chosen_object, chosen_move):
		new_location = []
		old_location = chosen_object.get_location()
		new_location.append(old_location[0] + chosen_move[0])
		new_location.append(old_location[1] + chosen_move[1])
		environment_size = self.environment.get_size()
		if new_location[0] == -1:
			new_location[0] = environment_size[0] -1

		elif new_location[0] == environment_size[0]:
			new_location[0] = 0

		if new_location[1] == -1:
			new_location[1] = environment_size[1] -1

		elif new_location[1] == environment_size[1]:
			new_location[1] = 0
		return new_location

class Environment:
	#Initialize environment of given size
	def __init__(self, size=[11,11]):
		self.size = size
		self.grid = [[ ' ' for i in range(0, size[0])] for y in range(0, size[1])]
	#Print the environment
	def print_grid(self):
		print "=========="
		for row in self.grid:
			print row
		print "=========="

	#Place an object at a given location in the environment
	def place_object(self, grid_object, new_location):
		self.grid[new_location[0]][new_location[1]] = grid_object

	def remove(self, location):
		self.grid[location[0]][location[1]] = ' '

	def get_size(self):
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
