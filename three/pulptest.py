import pulp
import sys

max_tables = 5
max_table_size = 4
guests = 'A B C D E F G I J K L M N O P Q R'.split()
total = {'A': 1, 'B': 2, 'C': -2, 'D': 0.1}
def happiness(action):
    """
    Find the happiness of the table
    - by calculating the maximum distance between the letters
    """
    print action
    print total[action]
    return total[action]
                
#create list of all possible tables
possible_keys = total.keys()

#create a binary variable to state that a table setting is used
x = pulp.LpVariable.dicts('key', possible_keys, 
                            lowBound = 0,
                            upBound = 1,
                            cat = pulp.LpInteger)
print x

seating_model = pulp.LpProblem("Wedding Seating Model", pulp.LpMinimize)

seating_model += sum([happiness(key) * x[key] for key in possible_keys])


seating_model.solve()

print "The choosen tables are out of a total of %s:"%len(possible_keys)
for table in possible_keys:
    if x[table].value() == 1.0:
        print table
