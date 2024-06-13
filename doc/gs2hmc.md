
# From grid search to HMC
This quick tutorial demonstrates how to revise the MTUQ grid search code to use SeisHMC for moment tensor inversion.
You can also review the following *.py files for code changes.
* **Double-Couple** solution: from [Grid Search](../examples/GridSearch.DoubleCouple.py) to [HMC](../examples/HMC.DoubleCouple.py). 
* **Full moment tensor** solution: from [Grid Search](../examples/GridSearch.FullMomentTensor.py) to [HMC](../examples/HMC.FullMomentTensor.py)

## Code changes

### Import the HMC solvers and plotting functions
* For double-couple solver
```python
from seishmc.DHMC.dc import DHMC_DC
from seishmc.visualization.viz_samples_dc import pairplot_samples_DC
```

* For full moment tensor solver
```python
from seishmc.DHMC.fmt import DHMC_FMT
from seishmc.visualization.viz_samples_fmt import pairplot_samples_FMT
```

### Specify the output directory
Specify the output directory to store the samples and results. Eg: 
```python
saving_dir = '../output/MT_inversion'
```

### Remove the moment tensor grid
We don't need to specify the moment tensor grid for HMC solver. 
The following code utilized in the grid search examples is **removed** in HMC examples:
```python
grid = DoubleCoupleGridRegular(
        npts_per_axis=40,
        magnitudes=[4.7, 4.8, 4.9])
```
and 
```python
grid = FullMomentTensorGridSemiregular(
        npts_per_axis=20,
        magnitudes=[4.7, 4.8, 4.9])
```

### Initialize HMC solver
The HMC solver must be initilized before using.

* for DC solver: 
```python
solver_hmc = DHMC_DC(misfit_bw, data_bw, greens_bw,
                     misfit_sw, data_sw, greens_sw,
                     saving_dir, b_save_cache=True,
                     n_step_cache=500, verbose=True)
```

* for FMT solver: 
```python
solver_hmc = DHMC_FMT(misfit_bw, data_bw, greens_bw,
                      misfit_sw, data_sw, greens_sw,
                      saving_dir, b_save_cache=True,
                      n_step_cache=500, verbose=True)
```
* If **b_save_cache** is True, the accepted samples will be saved at a step of **n_step_cache** at the **saveing_dir**.

### Set parameters for HMC solver
Those parameters need to be set for both DC and FMT solver. The default values in the examples can be utilized in future applications. 
```python
    # set the range of number of step
    solver_hmc.set_n_step(min=3, max=10)    # short chain, faster
    # solver_hmc.set_n_step(min=20, max=50)    # long chain, slower

    # set the range of step interval
    solver_hmc.set_epsilon(min=0.05, max=1.0)

    # set sigma_d, the default is 0.1 if comment.
    # solver_hmc.set_sigma_d(0.05)

    # set the number of accepted samples in total before stop
    n_sample = 1000
```

### Set initial solution
Arbitrary initial solutions are utilized in our examples. 
* For DC solver:
```python
    q0 = np.array([np.random.uniform(0, 360),
                   np.random.uniform(0, 90),
                   np.random.uniform(0, 180),
                   np.random.uniform(4.5, 5.0)])
    solver_hmc.set_q(q0)
```

* For FMT solver: 
```python
    q0 = np.array([np.random.uniform(0, 360),
                   np.random.uniform(0, 90),
                   np.random.uniform(0, 180),
                   np.random.uniform(4.5, 5.),
                   np.random.uniform(0, 180),
                   np.random.uniform(-30, 30)])
    solver_hmc.set_q(q0)
```


### Sampling
Label the sampling task with **task_id**. 'hmc01' will be utilized if not specified.
```python
task_id = 'HMC_task'
solver_hmc.sampling(n_sample=n_sample, task_id=task_id)
```
The sampling will stop after **n_sample** samples are accepted.


### Results and figures
SeisHMC provides functions to plot the accepted samples and marginal posterior distribution of source parameters.
* For DC solver
```python
pairplot_samples_DC(file_path=data_file, fig_saving_path=fig_path, init_sol=q0)
```

* For FMT solver
```python
pairplot_samples_FMT(file_path=data_file, fig_saving_path=fig_path, init_sol=q0)
```
Besides, SeisHMC also calls plotting functions in MTUQ to plot waveforms and beachballs.

