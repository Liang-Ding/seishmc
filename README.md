# SEISHMC
![SEISHMC](./doc/images/seishmc.png)

SEISHMC is a Python package that uses **Hamiltonian Monte Carlo** (HMC) sampling to invert the full moment tensor of earthquake sources.


## Installation

1. Requirements
    * [MTUQ](https://github.com/uafgeotools/mtuq) ([https://github.com/uafgeotools/mtuq](https://github.com/uafgeotools/mtuq))
    * Seaborn (>= 0.11.2) 
    * Numpy
    * Pandas


2. Install seisHMC

* For basic install:
```shell
git clone https://github.com/Liang-Ding/seishmc.git
cd seishmc
pip install -e .
```
* or using pip 
```shell
pip install seishmc
```

## Quick start
1. Double-Couple solution: [DC example](./examples/HMC.DoubleCouple.py)
2. Full Moment Tensor solution: [FMT example](./examples/HMC.FullMomentTensor.py)

