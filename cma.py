import numpy as np
from math import sqrt, log, exp

class StrategyMultiObjective(object):
    """Multiobjective CMA-ES strategy based on the paper [Voss2010]_. It
    is used similarly as the standard CMA-ES strategy with a generate-update
    scheme.

    :param population: An initial population of individual.
    :param sigma: The initial step size of the complete system.
    :param mu: The number of parents to use in the evolution. When not
               provided it defaults to the length of *population*. (optional)
    :param lambda_: The number of offspring to produce at each generation.
                    (optional, defaults to 1)
    :param indicator: The indicator function to use. (optional, default to
                      :func:`~deap.tools.hypervolume`)

    Other parameters can be provided as described in the next table

    +----------------+---------------------------+----------------------------+
    | Parameter      | Default                   | Details                    |
    +================+===========================+============================+
    | ``d``          | ``1.0 + N / 2.0``         | Damping for step-size.     |
    +----------------+---------------------------+----------------------------+
    | ``ptarg``      | ``1.0 / (5 + 1.0 / 2.0)`` | Taget success rate.        |
    +----------------+---------------------------+----------------------------+
    | ``cp``         | ``ptarg / (2.0 + ptarg)`` | Step size learning rate.   |
    +----------------+---------------------------+----------------------------+
    | ``cc``         | ``2.0 / (N + 2.0)``       | Cumulation time horizon.   |
    +----------------+---------------------------+----------------------------+
    | ``ccov``       | ``2.0 / (N**2 + 6.0)``    | Covariance matrix learning |
    |                |                           | rate.                      |
    +----------------+---------------------------+----------------------------+
    | ``pthresh``    | ``0.44``                  | Threshold success rate.    |
    +----------------+---------------------------+----------------------------+

    .. [Voss2010] Voss, Hansen, Igel, "Improved Step Size Adaptation
       for the MO-CMA-ES", 2010.

    """
    def __init__(self, no_variables, no_parents, no_offspring, sigma):
        self.dim = no_variables

        # Selection
        self.mu = no_parents
        self.lambda_ = no_offspring

        # Step size control
        self.d = 1.0 + self.dim / 2.0
        #self.ptarg = 1.0 / (5.0 + 0.5)
        self.ptarg = np.power((5 + (sqrt(1.0/2.0))), -1)
        self.cp = self.ptarg / (2.0 + self.ptarg)

        # Covariance matrix adaptation
        self.cc = 2.0 / (self.dim + 2.0)
        self.ccov = 2.0 / (self.dim ** 2 + 6.0)
        self.pthresh = 0.44

        # Internal parameters associated to the mu parent
        self.sigmas = [sigma] * no_parents
        # Lower Cholesky matrix (Sampling matrix)
        self.A = [np.identity(self.dim) for _ in range(no_parents)]
        # Inverse Cholesky matrix (Used in the update of A)
        self.invCholesky = [np.identity(self.dim) for _ in range(no_parents)]
        self.pc = [np.zeros(self.dim) for _ in range(no_parents)]
        self.psucc = [self.ptarg] * no_parents

        


    def generate(self, population, MIN_BOUND, MAX_BOUND):
        """Generate a population of :math:`\lambda` individuals of type
        *ind_init* from the current strategy.

        :param ind_init: A function object that is able to initialize an
                         individual from a list.
        :returns: A list of individuals with a private attribute :attr:`_ps`.
                  This last attribute is essential to the update function, it
                  indicates that the individual is an offspring and the index
                  of its parent.
        """
        arz = np.random.randn(self.lambda_, self.dim)
        individuals = list()

        ## Make sure every parent has a parent tag and index
        #for i, p in enumerate(self.parents):
         #   p._ps = "p", i

        # Each parent produce an offspring
        for i in range(self.lambda_):
            # print "Z", list(arz[i])
            
            sol = (population[i,:] + self.sigmas[i] * np.dot(self.A[i], arz[i]))
    
            individuals.append(sol)
            #individuals[-1]._ps = "o", i

        # Parents producing an offspring are chosen at random from the first front
      
        return individuals

    def _rankOneUpdate(self, invCholesky, A, alpha, beta, v):
        w = np.dot(invCholesky, v)

        # Under this threshold, the update is mostly noise
        if w.max() > 1e-20:
            w_inv = np.dot(w, invCholesky)
            norm_w2 = np.sum(w ** 2)
            a = sqrt(alpha)
            root = np.sqrt(1 + beta / alpha * norm_w2)
            b = a / norm_w2 * (root - 1)

            A = a * A + b * np.outer(v, w)
            invCholesky = 1.0 / a * invCholesky - b / (a ** 2 + a * b * norm_w2) * np.outer(w, w_inv)

        return invCholesky, A

    def update(self, chosen, ind_parent, parent_id, mu, parent_pop):
        """Update the current covariance matrix strategies from the
        *population*.

        :param population: A list of individuals from which to update the
                           parameters.
        """
        #chosen, not_chosen = self._select(population + self.parents)

        cp, cc, ccov = self.cp, self.cc, self.ccov
        d, ptarg, pthresh = self.d, self.ptarg, self.pthresh

        # Make copies for chosen offspring only
        last_steps = [self.sigmas[parent_id[ind]] if ind_parent[ind] == 0 else None for ind in range(mu)]
        sigmas = [self.sigmas[parent_id[ind]] if ind_parent[ind] == 0 else None for ind in range(mu)]
        invCholesky = [self.invCholesky[parent_id[ind]].copy() if ind_parent[ind] == 0 else None for ind in range(mu)]
        A = [self.A[parent_id[ind]].copy() if ind_parent[ind] == 0 else None for ind in range(mu)]
        pc = [self.pc[parent_id[ind]].copy() if ind_parent[ind] == 0 else None for ind in range(mu)]
        psucc = [self.psucc[parent_id[ind]] if ind_parent[ind] == 0 else None for ind in range(mu)]

        # Update the internal parameters for successful offspring
        for i in range(mu):
            t = ind_parent[i]
            p_idx = parent_id[i]

            # Only the offspring update the parameter set
            if t == 0:
                # Update (Success = 1 since it is chosen)
                psucc[i] = (1.0 - cp) * psucc[i] + cp
                sigmas[i] = sigmas[i] * exp((psucc[i] - ptarg) / (d * (1.0 - ptarg)))

                if psucc[i] < pthresh:
                    xp = np.array(chosen.variables[i,:])
                    x = np.array(parent_pop[p_idx,:])
                    #        pc = (1-cc)*pc + sqrt( cc * ( 2.-cc ) ) * c.m_lastStep;

                    pc[i] = (1.0 - cc) * pc[i] + sqrt(cc * (2.0 - cc)) * (xp - x) / last_steps[i]
                    invCholesky[i], A[i] = self._rankOneUpdate(invCholesky[i], A[i], 1 - ccov, ccov, pc[i])
                else:
                    pc[i] = (1.0 - cc) * pc[i]
                    pc_weight = cc * (2.0 - cc)
                    invCholesky[i], A[i] = self._rankOneUpdate(invCholesky[i], A[i], 1 - ccov + pc_weight, ccov, pc[i])

                self.psucc[p_idx] = (1.0 - cp) * self.psucc[p_idx] + cp
                self.sigmas[p_idx] = self.sigmas[p_idx] * exp((self.psucc[p_idx] - ptarg) / (d * (1.0 - ptarg)))

        # It is unnecessary to update the entire parameter set for not chosen individuals
        # Their parameters will not make it to the next generation
        
        l2 = parent_id[ind_parent == 0]
        l1 = np.array(range(mu))
        not_chosen = [x for x in l1 if x not in l2]
        #print len(not_chosen)

        for p_idx in not_chosen:
            self.psucc[p_idx] = (1.0 - cp) * self.psucc[p_idx]
            self.sigmas[p_idx] = self.sigmas[p_idx] * exp((self.psucc[p_idx] - ptarg) / (d * (1.0 - ptarg)))

        # Make a copy of the internal parameters
        # The parameter is in the temporary variable for offspring and in the original one for parents
        self.sigmas = [sigmas[ind] if ind_parent[ind] == 0 else self.sigmas[parent_id[ind]] for ind in range(mu)]
        self.invCholesky = [invCholesky[ind] if ind_parent[ind] == 0 else self.invCholesky[parent_id[ind]] for ind in range(mu)]
        self.A = [A[ind] if ind_parent[ind] == 0 else self.A[parent_id[ind]] for ind in range(mu)]
        self.pc = [pc[ind] if ind_parent[ind] == 0 else self.pc[parent_id[ind]] for ind in range(mu)]
        self.psucc = [psucc[ind] if ind_parent[ind] == 0 else self.psucc[parent_id[ind]] for ind in range(mu)]