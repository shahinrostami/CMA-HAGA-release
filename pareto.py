import numpy as np

def is_pareto_efficient(costs):
    """
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<=c, axis=1)  # Remove dominated points
    return is_efficient

def ranked_pareto_fronts(objectives):
    pareto_fronts = []
    sol_idx = np.array(range(0,objectives.shape[0]))
    temp_objectives = objectives[:]
    
    while(len(temp_objectives)):
        pareto_front = is_pareto_efficient(temp_objectives)
        pareto_fronts.append(sol_idx[pareto_front])
        temp_objectives = temp_objectives[pareto_front == False,:]
        sol_idx = sol_idx[pareto_front == False]
        
    return pareto_fronts