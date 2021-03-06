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
import matplotlib.pyplot as plt
import copy


class Game:
	def __init__(self, prey=None, predator=None, softmax=False, grid_size=[11,11], learning_type='Q-learning', agent_list=None, location_dict=None):
		""" Initalize environment and agents """
		print "Learning algorithm used in this episode: ", learning_type
		#Store the learning type
		self.learning_type = learning_type
		#Instantiate environment object with correct size, predator and prey locations
		self.environment = Environment(grid_size, location_dict)
		#Create prey if none was given
		if(prey==None):
			prey_policy = Policy(grid_size, prey=True, softmax=softmax, amount_agents=len(agent_list), agent_name='0', learning_type=learning_type)
			self.prey = Prey(prey_policy, str(0))
		#Else, store prey
		else:
			self.prey = prey
		#Create predator if none was given
		if(agent_list==None):
			predator_policy = Policy(grid_size, prey=False, softmax=softmax, amount_agents=1, agent_name='1', learning_type=learning_type)
			self.agent_list = [self.prey, Predator(predator_policy)]
		#Else, store the predator
		else:
			self.agent_list = agent_list



		print "Episode created with grid size ", grid_size

	def get_rounds(self, learning_rate, discount_factor, epsilon):
		""" Return rounds played """
		#Play rounds until the prey is caught, and return how many were needed
		rounds, caught, bumped = self.until_caught(learning_rate, discount_factor, epsilon)
		return rounds, self.agent_list, caught, bumped

	def until_caught(self, learning_rate, discount_factor, epsilon):
		""" Repeat turns until prey is caught. Returns number of steps until game stopped """
		steps = 0
		prey_caught = 0
		predators_bumped = 0
		#Runs turns until the prey is caught
		action_dict = None
		while(prey_caught == 0 and predators_bumped == 0):
			steps +=1
			#Get the current state
			state = self.environment.get_state()
			#Run turn and see if prey has been caught
			prey_caught, predators_bumped, action_dict = self.turn(state, learning_rate, discount_factor, epsilon, steps, action_dict)
			newstate = self.environment.get_state()
			#print "updated state: ", newstate
			#self.environment.print_grid()
		
		if prey_caught == True:
			print "Caught prey in " + str(steps) + " rounds!\n=========="
		elif predators_bumped == True:
			print "Predators bumped into each other in round " + str(steps) + "!\n=========="
		else:
			print "Game ended in " + str(steps) + " rounds!\n=========="
		
				
		#return steps, self.predator.get_policy_grid()
		return steps, prey_caught, predators_bumped

	
	def turn(self, old_state, learning_rate, discount_factor, epsilon, steps, action_dict=None):
		""" Plays one turn for prey and predator. Choose their action and adjust their state and location accordingly """
		#Get each agent's new location and move them within the environment
		# Copied old_state
		copy_old_state = copy.deepcopy(old_state)
		taken_actions = {}
		for agent in self.agent_list:
			if self.learning_type is not "SARSA":
				 agent_action, agent_move = agent.get_action(copy_old_state, epsilon)
			else:
				agent_name = agent.get_name()
				agent_move = action_dict.get(agent_name)[0]
				agent_action = action_dict.get(agent_name)[1]
			#Store the taken action, important for q-learning
			taken_actions[agent.get_name()] = agent_action
			# Prey trips with probability of 0.2
			if agent.get_name() == '0' and random.randint(1, 10) <= 2:
				agent_move = [0,0]
				#print 'THE PREY BROKE ITS LEG AND TRIPPED!'
				
			new_location = self.get_new_location(agent.get_name(), agent_move, grid_size=self.environment.grid_size)
			self.environment.move_object(agent.get_name(), new_location)
		#Retrieve the new state (location per agent)
		new_locations = self.environment.get_state()

		new_prey_location = new_locations['0']
		checked_location = []

		prey_caught = False
		predators_bumped = False

		for agent in new_locations.keys():
			#If agent is a predator
			if(agent != '0'):
				new_predator_location = new_locations[agent]
				#Check if this location is occupied by an already checked predator
				if(new_predator_location in checked_location):
					predators_bumped = True
				#Append to list of checked locations
				checked_location.append(new_locations[agent])
				if(new_locations[agent] == new_prey_location):
					prey_caught = True

		#Determine rewards for this turn:
		#If the predators bumped into eachother, they lose:
		if(predators_bumped):
			rewards_list = [-10 for x in range(0, len(self.agent_list)-1)]
			rewards_list = [10]+rewards_list
		#If the predators avoided eachother, and caught the prey, they win:
		elif(prey_caught):
			rewards_list = [10 for x in range(0, len(self.agent_list)-1)]
			rewards_list = [-10]+rewards_list
		else:
			rewards_list = [0 for x in range(0, len(self.agent_list))]

		new_action_dict = {}

		#If we're using q-learning, update the q-values using a greedy action in next state
		if(self.learning_type == 'Q-learning' or self.learning_type == 'Minimax'):
			s = copy_old_state
			s_prime = self.environment.get_state()
			for agent in self.agent_list:
				agent_name = agent.get_name()
				agent_action = taken_actions[agent_name]
		
				opponent_name = str((int(agent_name)+1)%2)
				opponent_action = taken_actions[opponent_name]
				
				#print 'all actions taken:', taken_actions
				#print "Agent ", agent.get_name(), " took action ", taken_actions[agent.get_name()]
				agent.q_learning(agent_action, opponent_action, s, s_prime, learning_rate, discount_factor, epsilon, self.agent_list, rewards_list, learning_type)
				#print "pol: ", agent.policy_grid.return_state_policy(s)
		elif(self.learning_type == 'SARSA'):
			s = copy_old_state
			s_prime = self.environment.get_state()
			for agent in self.agent_list:
				a = taken_actions[agent.get_name()]
				#print "Agent ", agent.get_name(), " took action ", taken_actions[agent.get_name()]
				action = agent.sarsa(a, s, s_prime, learning_rate, discount_factor, epsilon, self.agent_list, rewards_list)
				new_action_dict.update({agent.get_name():action})
				#print "pol: ", agent.policy_grid.return_state_policy(s)
		
		
		

		#Return caught or not
		return prey_caught, predators_bumped, new_action_dict


	def get_new_location(self, chosen_object, chosen_move, grid_size=[11,11]):
		""" Returns new location of an object when performs the chosen move """
		new_location = []
		#Retrieve the agent's position in the grid
		old_location = self.environment.get_location(chosen_object)
		#Get the size of the environment
		environment_size = self.environment.get_size()
		#Wrap edges to make grid toroidal
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



def reset_agents(location_dict, grid_size):
	#reset agents to original place on board!

	for agent in location_dict:
		if agent == "0":
			location_dict["1"] = [grid_size[0]/2,grid_size[1]/2]
		elif agent == "1":
			location_dict["0"] = [0,0]
		elif agent == "2":
			location = [grid_size[0]-1,grid_size[1]-1]
		elif agent == "3":
			location_dict["3"] = [grid_size[0]-1,0]
		elif agent == "4":
			location_dict["4"] = [0,grid_size[1]-1]

	return location_dict


def run_episodes(grid_size, N, learning_rate, discount_factor, epsilon, amount_predators=2, softmax=False, verbose=0, learning_type='Q-learning', experiments=5):
	""" Run N episodes and compute average """
	print 'Run Episodes: learning_type:', learning_type
	lose_y_list = []
	win_y_list = []
	y_list = []
	all_first_100_wins = 0
	all_last_100_wins = 0
	all_first_100_losses = 0
	all_last_100_losses = 0
	for y in range(0, experiments):
		print "initializing prey..."
		prey_pol = Policy(grid_size, amount_agents=amount_predators+1, agent_name='0', learning_type=learning_type)
		agent_list = [Prey(prey_pol, str(0))]
		#Prey has a name 0 and is in middle of the grid
		location_dict = {"0": [grid_size[0]/2,grid_size[1]/2]}
		for i in range(0, amount_predators):
			print "initializing predator ", i, "..."
			pred_pol = Policy(grid_size, amount_agents=amount_predators+1, agent_name=str(i+1), learning_type=learning_type)
			agent_list.append(Predator(pred_pol, str(i+1)))
			if(i == 0):
				location = [0,0]
			elif(i==1):
				location = [grid_size[0]-1,grid_size[1]-1]
			elif(i==2):
				location = [grid_size[0]-1,0]
			elif(i==3):
				location = [0,grid_size[1]-1]

			location_dict[str(i+1)] = location
		reset_dict = copy.deepcopy(location_dict)
		total_rounds = 0
		rounds_list = []
	#If we're using off-policy MC, initialize game/predator differently to allow separated learn/test runs
		game = Game(grid_size=grid_size, softmax=softmax, learning_type=learning_type, agent_list=agent_list, location_dict=location_dict)
		average_list = []
		counter=0
		current_rounds=0



		cumulative_losses = 0
		cumulative_wins = 0
		lose_list = []
		win_list = []
		first_100_wins = 0
		first_100_losses = 0
		last_100_wins = 0
		last_100_losses = 0
		for x in range(0, N):
			print "Round ", x, " in experiment ", y
			#print "Rounds needed to catch prey: ", current_rounds
			#Initialize episode
			#If we're using off-policy MC, initialize a learning and then a testing episode
			#current_rounds, policy_grid = game.get_rounds(learning_rate, discount_factor, epsilon)

			#TODO: Return agentlist and pass to next game
			current_rounds, agent_list, caught, bumped = game.get_rounds(learning_rate, discount_factor, epsilon)
			#print agent_list[0].policy_grid.return_state_policy(game.environment.get_state())
			#print agent_list[0].policy_grid.return_state_policy(game.environment.get_state())
			if(bumped):
				cumulative_losses +=1
				print "Rounds needed before predators bumped: ", current_rounds
				if x < 100:
					first_100_losses +=1
				elif N-x < 100:
					last_100_losses += 1
			elif(caught):
				cumulative_wins +=1
				print "Rounds needed before prey was caught: ", current_rounds
				if x < 100:
					first_100_wins +=1
				if N-x < 100:
					last_100_wins +=1
			win_list.append(cumulative_wins)
			lose_list.append(cumulative_losses)

			location_dict = reset_agents(reset_dict, grid_size)
			game = Game(grid_size=grid_size, softmax=softmax, learning_type=learning_type, agent_list=agent_list, location_dict=location_dict)

			#Add rounds needed in test episode to total_rounds	
			total_rounds += current_rounds
			#Add rounds needed in this episode to the list of rounds
			rounds_list.append(current_rounds)
		lose_y_list.append(lose_list)
		win_y_list.append(win_list)
		y_list.append(rounds_list)

		all_first_100_wins += first_100_wins
		all_last_100_wins += last_100_wins
		all_first_100_losses += first_100_losses
		all_last_100_losses += last_100_losses
		#Smooth graph
	av_wins = []
	av_losses = []
	av_rounds = []
	for number in range(0, len(lose_list)):
		yl_number = 0
		for yl in lose_y_list:
			yl_number += yl[number]
		av_losses.append(yl_number)

	for number in range(0, len(win_list)):
		yl_number = 0
		for yl in win_y_list:
			yl_number += yl[number]
		av_wins.append(yl_number)		

	#For every episode in every experiment
	for number in range(0, len(rounds_list)):
		yl_number = 0
		for yl in y_list:
			#Get this experiment's current episode
			yl_number += yl[number]
		av_rounds.append(yl_number/len(y_list))			


	#print "List of steps needed per episode: ", rounds_list

	#print "List of smoothed averages: ", average_rounds
	if(amount_predators>1):
		print "Av losses: ", av_losses
		print "Av wins: ", av_wins
	else:
		print "ylist: ", y_list
		print " rounds: ", av_rounds

	#Compute average number of rounds needed
	#average_rounds = float(average_rounds)/N
	#Compute list of variances
	#var_list = [(x-average_rounds)**2 for x in rounds_list]
	#Compute average variance
	#variance = float(sum(var_list)/len(var_list))
	#Compute standard deviation for N rounds
	#standard_deviation = math.sqrt(variance)
	#print "Average rounds needed over ", N, " episodes: ", average_rounds
	#print "Standard deviation: ", standard_deviation

	avg_first_100_wins = all_first_100_wins/experiments
	avg_last_100_wins = all_last_100_wins/experiments
	avg_first_100_losses = all_first_100_losses/experiments
	avg_last_100_losses = all_last_100_losses/experiments
	return av_wins, av_losses, av_rounds, avg_first_100_wins, avg_last_100_wins, avg_first_100_losses, avg_last_100_losses


if __name__ == "__main__":
        
        time_start = time.time()
    
	#Command line arguments
	parser = argparse.ArgumentParser(description="Run simulation")
	parser.add_argument('-runs', metavar='How many simulations should be run?', type=int)
	parser.add_argument('-discount', metavar='Specify the size of the discount factor for value iteration.', type=float)
	parser.add_argument('-verbose', metavar='Verbose level of game. 0: no grids/states, 1: only start and end, 2: all', type=int)
	parser.add_argument('-learning_rate', metavar='Specify value of learning rate', type=float)
	parser.add_argument('-epsilon', metavar='Specify value of epsilon', type=float)
	parser.add_argument('-grid_size', metavar='Specify grid size', type=int)
	parser.add_argument('-learning_type', metavar='Specify learning type', type=str)
	parser.add_argument('-softmax', metavar='Use softmax', type=str)
	parser.add_argument('-predators', metavar='Specify number of predators', type=int)
	parser.add_argument('-experiments', metavar='Average over number of experiments', type=int)
	args = parser.parse_args()

	#Default parameter values
	N = 5
	discount_factor = 0.9
	learning_rate = 0.5
	epsilon = 0.1
	grid_size = 11
	softmax = False	
	learning_type = "Q-learning"
	amount_predators = 1
	Y = 5

	#Command line parsing
	if(vars(args)['runs'] is not None):
		N = vars(args)['runs']
	if(vars(args)['experiments'] is not None):
		Y = vars(args)['experiments']
	if(vars(args)['learning_type'] is not None):
		learning_type = vars(args)['learning_type']
	if(vars(args)['predators'] is not None):
		amount_predators = vars(args)['predators']
	if(vars(args)['softmax'] is not None):
		if(vars(args)['softmax'] == "yes"):
			softmax = True
	if(vars(args)['grid_size'] is not None):
		grid_size = vars(args)['grid_size']
	if(vars(args)['discount'] is not None):
		discount_factor = vars(args)['discount']
	if(vars(args)['learning_rate'] is not None):
		learning_rate = vars(args)['learning_rate']
	if(vars(args)['epsilon'] is not None):
		epsilon = vars(args)['epsilon']
	if(vars(args)['verbose'] is not None):
		verbose = vars(args)['verbose']
	else:
		verbose = 2


	discount_factors = [0.2, 0.5, 0.7]
	learning_rates = [0.2, 0.5, 0.7]
	epsilon_rates = [0, 0.2, 0.5, 0.7, 0.9]

	avg_first_100_wins_list = []
	avg_last_100_wins_list = []
	avg_first_100_losses_list = []
	avg_last_100_losses_list = []
	# factor in discount_factors:	
	av_wins, av_losses, av_rounds, avg_first_100_wins, avg_last_100_wins, avg_first_100_losses, avg_last_100_losses = run_episodes([grid_size,grid_size], N, learning_rate, discount_factor, epsilon, amount_predators=amount_predators, softmax=softmax, learning_type=learning_type, experiments=Y)
	if(amount_predators == 1):
		plt.plot(av_rounds, label="rounds")
		plt.ylabel('Steps needed before catch')
	else:
		avg_first_100_wins_list.append(avg_first_100_wins)
		avg_last_100_wins_list.append(avg_last_100_wins)
		avg_first_100_losses_list.append(avg_first_100_losses)
		avg_last_100_losses_list.append(avg_last_100_losses)
		plt.plot(av_wins, label="wins")
		plt.plot(av_losses, label="losses")
	
		# Used to be in title: "Predators vs. prey "
		plt.ylabel('Predator average wins')

	title = str('2 predators vs. 1 prey -> learning_rate: ' + str(learning_rate) + 'discount factor' + str(discount_factor) + ' epsilon: ' + str(epsilon) + ' experiments: ' + str(Y) + ' learning type: ' + str(learning_type))
	#'predators: ' + str(amount_predators) +  ' gamma: ' + str(discount_factor)
#	title = "2 predators vs. 1 prey: discount factors"

	#print "RIGGED PREY!"
	print "first 100 wins: ", avg_first_100_wins_list
	print "last 100 wins: ", avg_last_100_wins_list
	print "first 100 losses: ", avg_first_100_losses_list
	print "last 100 losses: ", avg_last_100_losses_list
	plt.title(title)
	plt.legend()	
	plt.xlabel('Number of episodes')
	plt.show()


	print learning_type, 'ran for', N, 'runs,', Y, 'experiments and ', amount_predators, 'predators and finished in', time.time() - time_start, 'seconds'

