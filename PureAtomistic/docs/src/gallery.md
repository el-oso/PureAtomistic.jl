# Example gallery

Each page below is generated from a script in `PureAtomistic/docs/examples/` by the documentation
build. The scripts run no dynamics: they read the records in `PureAtomistic/bench/results/` (with
the commit, the date and the settings of each run), stored by `PureAtomistic/bench/md_validation.jl`
and, for the vacancy-diffusion page, by `PureAtomistic/bench/vacancy_analysis.jl`, and plot them.
Each script also runs on its own, from the directory where its figures should go:

```
julia --project=PureAtomistic/docs PureAtomistic/docs/examples/nve_conservation.jl
```

The animations on the pages are rendered ahead of time with GLMakie by
`PureAtomistic/docs/render_animations.jl` from the trajectories of the same runs, and committed in
`PureAtomistic/docs/src/assets/`. The validation runs use MACE-MPA-0, one Julia thread and one
BLAS thread; the vacancy-diffusion campaign runs MACE-MPA-0 in Float32 on the GPU (Metal).

| Page | Shows |
|:-----|:------|
| [Energy conservation](examples/nve_conservation.md) | total-energy drift and fluctuation of 500 Cu atoms over 2 ps at several time steps, and the largest stable one |
| [Canonical sampling](examples/thermostat_check.md) | the Bussi thermostat at 1200 K: mean and variance of the kinetic temperature against the canonical distribution |
| [A vacancy in hot Cu](examples/vacancy_hop_md.md) | one vacancy at 1300 K followed by the vacancy tracker, as an illustration |
| [Vacancy diffusion in Cu](examples/vacancy_diffusion.md) | the vacancy-diffusion campaign at 900 to 1200 K: hop rates, Arrhenius fit of the migration energy, comparison with the NEB barrier and thermal expansion, cell-size check and diffusion coefficients |
