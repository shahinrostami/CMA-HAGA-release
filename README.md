# CMA-HAGA-release
CMA-PAES-HAGA implementation in Python.

# Change log
- **v0.1** - **23-May-17** - first commit

# Recommended parameters
delta = 2
mu = 100, CMA-PAES-HAGA operates on small population sizes regardless of the number of objectives being considered.

# Relevant publications

```
@article{rostami2016covariance,
  title={Covariance matrix adaptation pareto archived evolution strategy with hypervolume-sorted adaptive grid algorithm},
  author={Rostami, Shahin and Neri, Ferrante},
  journal={Integrated Computer-Aided Engineering},
  volume={23},
  number={4},
  pages={313--329},
  year={2016},
  publisher={IOS Press}
}
```

```
@article{rostami2016fast,
  title={A fast hypervolume driven selection mechanism for many-objective optimisation problems},
  author={Rostami, Shahin and Neri, Ferrante},
  journal={Swarm and Evolutionary Computation},
  year={2016},
  publisher={Elsevier}
}
```

```
@inproceedings{rostami2012cma,
  title={Cma-paes: Pareto archived evolution strategy using covariance matrix adaptation for multi-objective optimisation},
  author={Rostami, Shahin and Shenfield, Alex},
  booktitle={Computational Intelligence (UKCI), 2012 12th UK Workshop on},
  pages={1--8},
  year={2012},
  organization={IEEE}
}
```