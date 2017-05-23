
# coding: utf-8

# ### Dr. Shahin Rostami
# ### http://www.shahinrostami.com
# @article{rostami2016covariance,
#   title={Covariance matrix adaptation pareto archived evolution strategy with hypervolume-sorted adaptive grid algorithm},
#   author={Rostami, Shahin and Neri, Ferrante},
#   journal={Integrated Computer-Aided Engineering},
#   volume={23},
#   number={4},
#   pages={313--329},
#   year={2016},
#   publisher={IOS Press}
# }

# ### Imports

# In[1]:

import copy
import random
import numpy as np
from collections import namedtuple

from optproblems import zdt
from optproblems import dtlz
from optproblems import wfg

from penalising_evaluator import evaluate
from cma import StrategyMultiObjective
from haga import select
get_ipython().magic('pdb')


# ### Visualisation

# In[2]:

VIZ = True

if(VIZ):
    import matplotlib.pyplot as plt
    from pandas.tools.plotting import parallel_coordinates
    import pandas as pd
    from IPython.display import display, clear_output
    from mpl_toolkits.mplot3d import Axes3D


# ### Data Structures

# In[3]:

Population = namedtuple("Population", "variables objectives")


# ### Problem Configuration

# In[4]:

M = 4 # no. objectives
V = 12 # no. variables

#k = 2 * (M - 1)
#problem = wfg.WFG1(M, V, k) # create instance of problem

problem = dtlz.DTLZ2(M,V)  # NOTE: some test function implementations are broken in optproblems package.
#problem = zdt.ZDT4()

# variable boundaries
bounds_lower = problem.min_bounds
bounds_upper = problem.max_bounds


# ### Algorithm Configuration

# In[5]:

delta = 2 # divisions per objective

max_gens = 1000 # max number of generations

pen_alpha = 1.0 / V / M # how much we penalize infeasible solutions

mu = 100 # population size
mu_pop = Population(variables=np.zeros((mu,V)), objectives=np.zeros((mu,M)))
lambda_pop = Population(variables=np.zeros((mu,V)), objectives=np.zeros((mu,M)))

cma = StrategyMultiObjective(V, mu, mu, 1)

max_refs = np.zeros(M)


# ### Initialise Population

# In[ ]:

for n in range(0, mu):
    solution = np.zeros(V)
    for v in range(0, V):
        solution[v] = np.random.uniform(bounds_lower[v], bounds_upper[v], size=1)
    mu_pop.variables[n, :] = solution

mu_pop = evaluate(problem, mu, mu_pop,bounds_lower, bounds_upper, pen_alpha)


# ### Generational Loop

# In[ ]:

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1) 

for g in range(max_gens):
    
    lambda_pop = Population(variables=np.array(cma.generate(mu_pop.variables, bounds_lower, bounds_upper)), objectives=np.zeros((mu,M)))
    lambda_pop = evaluate(problem, mu, lambda_pop,bounds_lower, bounds_upper, pen_alpha)

    max_refs = np.maximum(max_refs, np.concatenate((mu_pop.objectives, lambda_pop.objectives)).max(axis=0))

    cma_parent_pop = mu_pop.variables[:]

    (parent_id, is_parent, mu_pop) = select(mu_pop, lambda_pop, mu, delta, max_refs)

    cma.update(mu_pop, is_parent, parent_id, mu, cma_parent_pop)
   
    # VISUALISATION
    if(VIZ):
        if(M == 3):
            if((g % 10) == 0):
                ax = fig.add_subplot(1, 1, 1,projection='3d') 
                ax.cla()
                ax.view_init(30, 90)


                ax.scatter(mu_pop.objectives[:,0], mu_pop.objectives[:,1], mu_pop.objectives[:,2], marker='o', depthshade=True,alpha = 1,linewidth='0.5')


                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_zlim(0, 1)
                plt.title("{} {}".format(g, max_refs))

                display(fig)


                clear_output(wait = True)
        else:
            df_lambda_pop = pd.DataFrame(lambda_pop.objectives)
            df_lambda_pop['Generation'] = np.zeros(lambda_pop.objectives.shape[0])
            df_mu_pop = pd.DataFrame(mu_pop.objectives)
            df_mu_pop['Generation'] = np.zeros(mu_pop.objectives.shape[0]) + 1

            result = pd.concat([df_mu_pop])

            if((g % 10) == 0):
                ax.cla()
                parallel_coordinates(result,'Generation')
                plt.title("{} {}".format(g, max_refs))
                #plt.ylim((0, 3))
                display(fig)
                clear_output(wait = True)


# In[ ]:

np.savetxt("mu.csv", mu_pop.objectives, delimiter=",")


# In[ ]:




# In[ ]:



