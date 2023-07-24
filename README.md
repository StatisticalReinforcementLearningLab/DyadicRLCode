# Dyadic Reinforcement Learning
This repository contains the code to reproduce the numerical results in Li et al. (2023).

## Folders
* maze_simulation/: This contains the code to reproduce the numerical results from Section 4. Each subfolder contains the code for one figure or a group of figures. For instance, the folder "env1_dense/" contains the "regret.pdf" file, which corresponds to Figure 5(b) in the paper. The ".py" and ".sh" files within this folder can be run to generate this plot.
* test_bed/: This contains the code to reproduce the results from Section 5 (the simulation test bed). The "preprocessing/" subfolder contains R code for data preprocessing. The "test_algorithm/" subfolder contains the code for implementing reinforcement learning algorithms. Each subfolder therein contains the code for one figure or a group of figures.

## Replicating the experiments
* To replicate the experiments, one needs to run the "run-me_dyad.sh" file on a computing cluster, followed by the "make_plot.py" or "make_heatmap.py" file to generate the plot. The "script_dyad.sh" file is specifically designed for the [FAS RC](https://www.rc.fas.harvard.edu/) cluster at Harvard. You may need to modify these files as per your requirements.

## References
Shuangning Li, Lluís Salvat Niell, Sung Won Choi, Inbal Nahum-Shani, Guy Shani and Susan A. Murphy. <b>Dyadic Reinforcement Learning</b>. 2023. [[arXiv](http://people.seas.harvard.edu/~samurphy/)]
