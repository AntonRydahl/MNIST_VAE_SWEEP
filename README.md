#Parameter Search and Profiling for Variational Autoencode Example

## Profiling
From the initial profiling it was found that almost 50 % of the time was spend on loading and transforming the data.
<p align="center">
  <img src="./reports/figures/before_improving_prof.png" width="500" title="hover text">
</p>
Thus it was chosen to separate this into two different steps. After transforming and storing the dataset in `data/processed` we obtained the following results:
<p align="center">
  <img src="./reports/figures/after_improving_prof.png" width="500" title="hover text">
</p>
The two originally profiled files can be found in `src/legacy_models`.

## Hyperparameter Tuning using WandB
