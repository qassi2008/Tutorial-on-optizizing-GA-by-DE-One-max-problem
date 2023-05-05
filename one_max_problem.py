# inspiration code https://deap.readthedocs.io/en/master/examples/ga_onemax.html
import random
import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

# Define the problem to optimize One-max problem (maximize the number of ones in a binary string)
def fitness_function(individual):
   return sum(individual),

# configuration of Genetic Algorithm
INDIVIDUAL_SIZE = 50
POPULATION_SIZE = 50
MAX_GENERATIONS = 100

# configuration of Differential Evolution
DE_MAX_GENERATIONS = 20
DE_POPULATION_SIZE = 20
DE_CROSSOVER = 0.5
DE_F = 0.5  # Scaling factor

# Define fitness and individuals
creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.Fitness)

toolbox = base.Toolbox()

# Define the GA problem
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=INDIVIDUAL_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Define GA operators
toolbox.register("evaluate", fitness_function)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# Create an initial population and evaluate it
initial_population = toolbox.population(n=POPULATION_SIZE)
fitnesses = list(map(toolbox.evaluate, initial_population))
for ind, fit in zip(initial_population, fitnesses):
   ind.fitness.values = fit

# Identify the best individual in the initial population
best_initial_individual = tools.selBest(initial_population, 1)[0]
print("Fitness of the best individual before DE optimization: {:.2f}".format(best_initial_individual.fitness.values[0]))

# DE optimization of GA parameters
def de_parameter_optimization():
   def de_fitness_function(params):
       mut_prob, sel_pressure = params
       pop = toolbox.population(n=POPULATION_SIZE)
       hof = tools.HallOfFame(1)
       stats = tools.Statistics(lambda ind: ind.fitness.values)
       stats.register("avg", np.mean)
       stats.register("min", min)
       stats.register("max", max)

       _, _ = algorithms.eaSimple(pop, toolbox, cxpb=1.0, mutpb=mut_prob, ngen=MAX_GENERATIONS, stats=stats, halloffame=hof,
                                  verbose=False)

       return hof[0].fitness.values[0],

   creator.create("DEFitnessMax", base.Fitness, weights=(1.0,))
   creator.create("DEIndividual", list, fitness=creator.DEFitnessMax)

   de_toolbox = base.Toolbox()
   de_toolbox.register("mut_prob", random.uniform, 0.1, 0.9)
   de_toolbox.register("sel_pressure", random.uniform, 1.0, 2.0)
   de_toolbox.register("individual", tools.initCycle, creator.DEIndividual,
                       (de_toolbox.mut_prob, de_toolbox.sel_pressure), n=1)
   de_toolbox.register("population", tools.initRepeat, list, de_toolbox.individual)
   de_toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=[0.1, 1.0], up=[0.9, 2.0], eta=20.0)
   de_toolbox.register("mutate", tools.mutPolynomialBounded, low=[0.1, 1.0], up=[0.9, 2.0], eta=20.0, indpb=0.5)
   de_toolbox.register("select", tools.selBest)
   de_toolbox.register("evaluate", de_fitness_function)

   pop = de_toolbox.population(n=DE_POPULATION_SIZE)
   hof = tools.HallOfFame(1)
   stats = tools.Statistics(lambda ind: ind.fitness.values)
   stats.register("avg", np.mean)
   stats.register("min", min)
   stats.register("max", max)

   _, _ = algorithms.eaSimple(pop, de_toolbox, cxpb=DE_CROSSOVER, mutpb=1 - DE_CROSSOVER, ngen=DE_MAX_GENERATIONS,
                              stats=stats, halloffame=hof, verbose=True)

   return hof[0]

   # Optimize GA parameters using DE
best_params = de_parameter_optimization()
print("Best parameters found by DE: Mutation probability = {:.2f}, Selection pressure = {:.2f}".format(*best_params))

# Run GA with optimized parameters
toolbox.register("mutate", tools.mutFlipBit, indpb=best_params[0])
toolbox.register("select", tools.selTournament, tournsize=int(best_params[1]))

pop = toolbox.population(n=POPULATION_SIZE)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", min)
stats.register("max", max)

logbook = tools.Logbook()


# Custom eaSimple function with logbook recording

def custom_eaSimple (population, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=__debug__):

   logbook = tools.Logbook()
   logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

   # Evaluate the individuals with an invalid fitness
   invalid_ind = [ind for ind in population if not ind.fitness.valid]
   fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
   for ind, fit in zip(invalid_ind, fitnesses):
       ind.fitness.values = fit

   if halloffame is not None:
       halloffame.update(population)

   record = stats.compile(population) if stats is not None else {}
   logbook.record(gen=0, nevals=len(invalid_ind), **record)
   if verbose:
       print(logbook.stream)

   # Begin the generational process
   for gen in range(1, ngen + 1):
       offspring = toolbox.select(population, len(population))
       offspring = list(offspring)

       # Apply crossover and mutation on the offspring
       for i in range(1, len(offspring), 2):
           if random.random() < cxpb:
               offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
               del offspring[i - 1].fitness.values, offspring[i].fitness.values

       for i in range(len(offspring)):
           if random.random() < mutpb:
               offspring[i] = toolbox.mutate(offspring[i])[0]
               del offspring[i].fitness.values

       # Evaluate the individuals with an invalid fitness
       invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
       fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
       for ind, fit in zip(invalid_ind, fitnesses):
           ind.fitness.values = fit

       # Replace the old population by the offspring
       population[:] = offspring

       if halloffame is not None:
           halloffame.update(population)

       # Append the current generation statistics to the logbook
       record = stats.compile(population) if stats is not None else {}
       logbook.record(gen=gen, nevals=len(invalid_ind), **record)
       if verbose:
           print(logbook.stream)

   return population, logbook
pop, logbook = custom_eaSimple(pop, toolbox, cxpb=1.0, mutpb=best_params[0], ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)
print("Best individual found by GA: {}".format(hof[0]))
print("Fitness of the best individual after DE optimization: {:.2f}".format(hof[0].fitness.values[0]))

# Plot the logbook statistics
gen = logbook.select("gen")
avg = logbook.select("avg")
min_ = logbook.select("min")
max_ = logbook.select("max")

plt.figure()
plt.plot(gen, avg, label="Average")
plt.plot(gen, min_, label="Minimum")
plt.plot(gen, max_, label="Maximum")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend(loc="lower right")
plt.title("GA Fitness over Generations")
plt.grid()
plt.show()