'''
Dr. Shahin Rostami
http://www.shahinrostami.com
@article{rostami2016fast,
  title={A fast hypervolume driven selection mechanism for many-objective optimisation problems},
  author={Rostami, Shahin and Neri, Ferrante},
  journal={Swarm and Evolutionary Computation},
  year={2016},
  publisher={Elsevier}
}
'''
import numpy as np
from pareto import ranked_pareto_fronts
from chv import chv_selection
from chv import worst_chv
from collections import namedtuple

Population = namedtuple("Population", "variables objectives")

def select(parent_population, offspring_population, mu, delta, ref):
    merged_objectives = np.concatenate((parent_population.objectives, offspring_population.objectives), axis=0)
    merged_variables = np.concatenate((parent_population.variables, offspring_population.variables), axis=0)
    merged_is_parent = np.concatenate((np.ones(mu), np.zeros(mu)), axis=0).astype(int)
    merged_id = np.concatenate((range(mu), range(mu)), axis=0).astype(int)

    # remove duplicates
    a = merged_objectives
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx = np.unique(b, return_index=True)

    merged_objectives = merged_objectives[idx,:]
    merged_variables = merged_variables[idx,:]
    merged_is_parent = merged_is_parent[idx]
    merged_id = merged_id[idx]
    
    ranked_idx = ranked_pareto_fronts(merged_objectives)  
    sel_idx = np.array([]).astype(int)
    
    for idx in ranked_idx:
        if(len(sel_idx) < mu):
            if(len(sel_idx) + len(idx) <= mu):
                sel_idx = np.append(sel_idx,idx)
            else:
                hv_pop = merged_objectives[idx,:]
                #ref = (hv_pop.max(axis=0) * .1) + hv_pop.max(axis=0)
                ref = (ref * .1) + ref
                rejected = []
                if(delta == 1):
                    rejected = chv_selection(hv_pop, ref, (mu - len(sel_idx)))
                else:
                    rejected = haga_selection(hv_pop, ref, (mu - len(sel_idx)), delta)
                temp_idx = idx[:]
                temp_idx = np.delete(temp_idx, rejected,axis=0)
                sel_idx = np.append(sel_idx,temp_idx)



                
    merged_objectives = merged_objectives[sel_idx,:]
    merged_variables = merged_variables[sel_idx,:]
    merged_is_parent = merged_is_parent[sel_idx]
    merged_id = merged_id[sel_idx]
    merged_pop = Population(variables=merged_variables, objectives=merged_objectives)
    
    return (merged_id, merged_is_parent, merged_pop)

def haga_selection(original_pop, ref, cap, delta):
    lambda_pop = original_pop[:]
    popsize = original_pop.shape[0]
     
    nobj = lambda_pop.shape[1]

    original_pop = lambda_pop[:]

    grid_max = np.amax(lambda_pop, axis=0)
    grid_min = np.amin(lambda_pop, axis=0)
    extremes = grid_min[:]

    grid_range = abs(grid_max - grid_min)
    grid_pad = grid_range * .1

    grid_max = grid_max + grid_pad
    grid_min = grid_min - grid_pad

    grid_range = abs(grid_max - grid_min)
    grid_step = grid_range / delta

    # remove (preserve) extreme solutions
    extreme_idx = np.array([]).astype(int)

    worst_idx = np.array([]).astype(int)


    for y in range(lambda_pop.shape[1]):
        min_idx = np.where(lambda_pop[:,y] == extremes[y])
        extreme_idx = np.append(extreme_idx, min_idx[0][0])

    extreme_idx = np.unique(extreme_idx)
    if(len(extreme_idx) >= cap):
        for r in range(len(extreme_idx) - cap):
            rejected_pop = original_pop[extreme_idx]
            rejected = worst_chv(rejected_pop, ref, nobj, (len(extreme_idx)))
            extreme_idx = np.delete(extreme_idx, (rejected), axis=0)

        best_idx = extreme_idx
        l1 = np.array(range(len(original_pop)))
        worst_idx = [x for x in l1 if x not in best_idx]
    else:

        grid_locations = np.zeros(lambda_pop.shape).astype(int)
        for y in range(lambda_pop.shape[0]):
            grid_locations[y,:] = np.floor_divide((lambda_pop[y,:] - grid_min), grid_step) + 1 

        # grid_density
        b = np.ascontiguousarray(grid_locations).view(np.dtype((np.void, grid_locations.dtype.itemsize * grid_locations.shape[1])))
        unique_a, grid_density = np.unique(b, return_counts=True)
        unique_a = unique_a.view(grid_locations.dtype).reshape(-1, grid_locations.shape[1])

        # target grid density
        ideal_grid_pop_size = cap / len(grid_density)

        sel_grid = 0
        while(sum(grid_density) > cap):
            if(grid_density[sel_grid] == max(grid_density)):
                grid_density[sel_grid] = grid_density[sel_grid] - 1
            sel_grid = (sel_grid+1) % len(grid_density)

        init_idx = np.array([]).astype(int)
        rejected = []
        for s in range(len(grid_density)):
            grid_sel = np.where(np.all(grid_locations==unique_a[s],axis=1))[0]
            grid_pop = lambda_pop[grid_sel]
            for r in range(len(grid_sel) - grid_density[s]):
                grid_rejected = worst_chv(grid_pop, ref, nobj, (len(grid_pop)))
                rejected.append(grid_sel[grid_rejected])
                grid_sel = np.delete(grid_sel, (grid_rejected), axis=0)
                grid_pop = np.delete(grid_pop, (grid_rejected), axis=0)
        
        for s in range(len(rejected)):
            reject = np.where(np.all(original_pop==lambda_pop[rejected[s]],axis=1))
            worst_idx = np.append(worst_idx,reject)
            


    return worst_idx

