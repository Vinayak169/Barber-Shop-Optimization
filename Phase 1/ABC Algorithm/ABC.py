import numpy as np
import random
import matplotlib.pyplot as plt
import itertools

#Specific parameters
n_customers = 80         # no. of customers at a particular day
n_barbers = 3            # no. of barbers available in shop
n_operations = 4         # no. of operations can be performed
max_obj_var_per = 5      # maximum variance we can except in sense of percentage of objective

class Barber_shop:
    def __init__(self, n_customers, n_barbers, operation_list, customer_type, schedule, due_time):
        self.n_customers = n_customers
        self.n_barbers = n_barbers
        self.operation_list = operation_list
        self.customer_type = customer_type
        self.schedule = schedule
        self.due_time = due_time
        self.operation_costs = [100, 50, 150, 200]
        self.weight = []
        self.time_required = []
        self.price = []
        self.membership = 200


    def get_weight(self):
        for ctype in self.customer_type:
            if ctype == 'Standard':
                self.weight.append(1)
            else:
                self.weight.append(2)

    def get_operation_price(self):
        for operation in self.operation_list:
            price_j = np.dot(operation, self.operation_costs)
            self.price.append(price_j)

    def normalvariate(self, mu, sigma):
        return random.normalvariate(mu, sigma)

    def get_operation_time(self):
        for operation in self.operation_list:
            operation_times = [round(self.normalvariate(10,2)), round(self.normalvariate(5,1)), 
                                round(self.normalvariate(15,3)), round(self.normalvariate(20,5))]
            time_j = np.dot(operation, operation_times)
            self.time_required.append(time_j)

    def initialize(self):
        self.get_weight()
        self.get_operation_price()
        self.get_operation_time()

    def get_barber(self, j, schedule):
        for n in range(len(schedule)):
            for m in range(len(schedule[n])):
                if schedule[n][m] == j:
                    return n

    def get_start_time(self, j, schedule, time_required):
        i = self.get_barber(j, schedule)
        s_i = schedule[i]
        start_j=0
        for k in s_i:
            if k==j:
                break
            start_j+=time_required[k]
        return start_j

    def get_discount(self, j, schedule, time_required):
        start_time = self.get_start_time(j, schedule, time_required)
        return 0.2 if start_time + time_required[j] > self.due_time[j] else 0

    def objective(self, schedule, time_required):
        net_profit = 0
        for j in range(self.n_customers):
            wj = self.weight[j]
            pj = self.get_discount(j, schedule, time_required)
            cj = self.price[j]
            net_profit += (1-wj*pj)*cj + (wj-1)*self.membership
        return net_profit

#Generating random operations for each customers
def generate_operations(n_customers, n_operations):
    operation_list = []              # list containing which operations a particular customer wants
    for _ in range(n_customers):
        op = [0 for _ in range(n_operations)]
        while op == [0 for _ in range(n_operations)]:        # to ensure each customer want at least one operation
            op = [random.randint(0,1) for _ in range(n_operations)]      # randomly assign operations 
        operation_list.append(op)
    return operation_list

def generate_customer_type(n_customers):
    std = int(0.7*n_customers)     # considering 70% are normal customers at 200 member fees
    type_list = ['Standard' for _ in range(std)]
    type_list.extend('Premium' for _ in range(n_customers - std))
    random.shuffle(type_list)      # to make list i.e. customer random
    return type_list

def generate_due_time(n_customers, n_barbers):
    return [random.randint(50, int(10*n_customers/n_barbers)) for _ in range(n_customers)]

# Define problem parameters
max_iterations = 1000
num_employed_bees = 80
num_onlooker_bees = 80
num_scout_bees = 80
max_trials = 50
trial = [0 for _ in range(num_employed_bees)]

# Define function to generate a new solution by swapping two integer values
def generate_new_solution(bee, random_bee):
    new_solution = bee
    
    for _ in range(2):
        sublist_index = random.randrange(len(new_solution))
        position = random.randrange(len(new_solution[sublist_index]))
    
        new_customer_no = random_bee[sublist_index][position]
    
        n, m = 0, 0
        for i in range(len(new_solution)):
            for j in range(len(new_solution[i])):
                if new_solution[i][j] == new_customer_no:
                    n, m = i, j
    
        new_solution[sublist_index][position], new_solution[n][m] = new_solution[n][m], new_solution[sublist_index][position]
    return new_solution

# Define function to generate a new solution by randomly assigning integer values to the tasks and resources
def generate_random_solution():
    solution = [[] for _ in range(n_barbers)]
    orders = list(range(n_customers))
    random.shuffle(orders)
    for i, num in enumerate(orders):
        solution[i % n_barbers].append(num)
    return solution

operation_list = generate_operations(n_customers, n_operations)
customer_type = generate_customer_type(n_customers)
due_time = generate_due_time(n_customers, n_barbers)
random_sol = generate_random_solution()

shop_init = Barber_shop(n_customers, n_barbers, operation_list, customer_type, random_sol, due_time)  # initialise problem parameters
shop_init.initialize()       # get weight, operation price and operation time

# Initialize the employed bees, their fitness values, and the best solution found so far
employed_bees = [generate_random_solution() for _ in range(num_employed_bees)]
employed_bees_obj = [shop_init.objective(bee, shop_init.time_required) for bee in employed_bees]
def fitness(bee):
    return 1/(1 + shop_init.objective(bee, shop_init.time_required))
employed_bees_fitness = [fitness(bee) for bee in employed_bees]
best_fitness = min(employed_bees_fitness)
def best_sol(best_fitness):
    best_solution = []
    for i in range(len(employed_bees_fitness)):
        if employed_bees_fitness[i] == best_fitness:
            best_solution = employed_bees[i]
    return best_solution
best_solution = best_sol(best_fitness)
Objective = []
Iteration_no = []

# Main loop
for iteration in range(max_iterations):
    # Employed bees phase
    for i in range(num_employed_bees):
        random_bee = random.choice(employed_bees)
        new_solution = generate_new_solution(employed_bees[i], random_bee)
        new_fitness = fitness(new_solution)
        if new_fitness < employed_bees_fitness[i]:
            employed_bees[i] = new_solution
            employed_bees_fitness[i] = new_fitness
            trial[i] = 0
        else:
            trial[i] += 1

    # Onlooker bees phase
    onlooker_bees = employed_bees[:]
    onlooker_bees_fitness = employed_bees_fitness[:]
    max_fitness = max(onlooker_bees_fitness)
    for b, i in itertools.product(range(80), range(num_onlooker_bees)):
        probability = 0.9*(employed_bees_fitness[i] / max_fitness) + 0.1
        rand_probability = random.random()
        if rand_probability < probability:
            random_bee = random.choice(onlooker_bees)
            new_solution = generate_new_solution(onlooker_bees[i], random_bee)
            new_fitness = fitness(new_solution)
            if new_fitness < employed_bees_fitness[i]:
                onlooker_bees[i] = new_solution
                onlooker_bees_fitness[i] = new_fitness
                trial[i] = 0
            else:
                trial[i] += 1
        else:
            trial[i] += 1

    # Scout bees phase
    scout_bees = []
    scout_bees_fitness = []
    for i in range(num_scout_bees):
        if trial[i] >= max_trials:
            scout_bees.append(generate_random_solution())
            scout_bees_fitness.append(fitness(scout_bees[i]))
            trial[i] = 0
        else:
            scout_bees.append(onlooker_bees[i])
            scout_bees_fitness.append(onlooker_bees_fitness[i])

    # Update employed bees and best solution
    for i in range(num_employed_bees):
        if onlooker_bees_fitness[i] < employed_bees_fitness[i]:
            employed_bees[i] = onlooker_bees[i]
            employed_bees_fitness[i] = onlooker_bees_fitness[i]
        if scout_bees_fitness[i] < employed_bees_fitness[i]:
            employed_bees[i] = scout_bees[i]
            employed_bees_fitness[i] = scout_bees_fitness[i]
        if employed_bees_fitness[i] < best_fitness:
            best_solution = employed_bees[i]
            best_fitness = employed_bees_fitness[i]

    Objective.append(shop_init.objective(best_solution, shop_init.time_required))
    Iteration_no.append(iteration + 1)
    print(shop_init.objective(best_solution, shop_init.time_required))
    # Print progress
    print("Iteration:", iteration+1, "Best fitness:", best_fitness, "Best_solution:", best_solution)

Best_output = shop_init.objective(best_solution, shop_init.time_required)
print(Best_output)
print(Objective)

plt.plot(Iteration_no, Objective)
plt.show()