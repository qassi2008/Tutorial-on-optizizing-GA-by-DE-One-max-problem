# Tutorial-on-optizizing-GA-by-DE-One-max-problem
Genetic Algorithm with Differential Evolution Optimization for One-max Problem
This repository contains a Python script that combines a Genetic Algorithm (GA) with Differential Evolution (DE) to solve the One-max problem. The goal of the One-max problem is to maximize the number of ones in a binary string. The script uses the DEAP (Distributed Evolutionary Algorithms in Python) library to implement both GA and DE.

Problem Description
The One-max problem is an optimization problem that aims to find a binary string with the maximum number of ones. The provided script uses GA to search for the optimal solution and DE to optimize the GA parameters, such as mutation probability and selection pressure.

Dependencies
To run the script, you need to have the following dependencies installed:

Python 3.x
DEAP
NumPy
Matplotlib
You can install them using pip:

Copy code
pip install deap numpy matplotlib
Usage
To run the script, simply execute the following command in your terminal or command prompt:

Copy code
python ga_de_onemax.py
The script will first run GA with an initial population and then optimize the GA parameters using DE. After that, it will run GA again with the optimized parameters and display the best individual found. Finally, the script will plot the GA fitness over generations.

Output
The script will output:

Fitness of the best individual before DE optimization.
Best parameters found by DE (mutation probability and selection pressure).
Best individual found by GA.
Fitness of the best individual after DE optimization.
A plot of GA fitness over generations, showing the average, minimum, and maximum fitness.
