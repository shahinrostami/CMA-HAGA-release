import numpy as np
from optproblems import Individual
from collections import Sequence

def distance(feasible_ind, original_ind):
    """A distance function to the feasibility region."""
    return sum((f - o)**2 for f, o in zip(feasible_ind, original_ind))

def closest_feasible(individual, MIN_BOUND, MAX_BOUND):
    """A function returning a valid individual from an invalid one."""
    feasible_ind = np.array(individual)
    feasible_ind = np.maximum(MIN_BOUND, feasible_ind)
    feasible_ind = np.minimum(MAX_BOUND, feasible_ind)
    return feasible_ind

def valid(individual, MIN_BOUND, MAX_BOUND):
    """Determines if the individual is valid or not."""
    if any(individual < MIN_BOUND) or any(individual > MAX_BOUND):
        return False
    return True

def wrapper(problem, individual, MIN_BOUND, MAX_BOUND, pen_alpha):    
    fitness_weights = np.empty(problem.num_objectives)
    fitness_weights.fill(-1.0)
    
    f_ind = closest_feasible(individual, MIN_BOUND, MAX_BOUND)
    solution = Individual(phenome=f_ind)
    problem.evaluate(solution)
    f_fbl = solution.objective_values[:]
    weights = tuple(1.0 if w >= 0 else -1.0 for w in fitness_weights)

    if len(weights) != len(f_fbl):
        raise IndexError("Fitness weights and computed fitness are of different size.")

    dists = tuple(0 for w in fitness_weights)
    dist = distance(f_ind, individual)

    if not isinstance(dists, Sequence):
        dists = repeat(dists)

    pen_fbl = tuple(f - w * pen_alpha * dist for f, w, d in zip(f_fbl, weights, dists))

    return pen_fbl

def evaluate(problem, mu, population, bounds_lower, bounds_upper, pen_alpha):
    for n in range(0, mu):
        individual = population.variables[n,:]
        if valid(individual,bounds_lower, bounds_upper):
            
            solution = Individual(phenome=individual)
            problem.evaluate(solution)
        else:
            solution = wrapper(problem, individual,bounds_lower, bounds_upper, pen_alpha)
            solution = Individual(objective_values=solution)
            
        population.objectives[n,:] = solution.objective_values[:]
    return population