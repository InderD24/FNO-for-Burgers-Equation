# Fourier Neural Operator for viscous Burgers' equation

Learn the map from an initial periodic field to its evolved solution:

$$u_t + (u^2/2)_x = \nu u_{xx},\quad x\in[-1,1),\quad u(-1,t)=u(1,t).$$

This project generates nonlinear PDE targets with a checked numerical solver,
trains a standard 1D Fourier Neural Operator (FNO) and a periodic CNN baseline,
and evaluates both against held-out solutions. It also compares unchanged-input
and heat-diffusion baselines, checks physical behavior, and tests finer grids
without retraining.

**Correction to the original experiment:** the old notebook generated targets by
multiplying a sine wave by `exp(-nu*pi**2*T)`. That is heat diffusion, not nonlinear
Burgers evolution. Its saved R² score does not apply to this experiment. The new
notebook uses the same package and configuration as the command-line pipeline.

## Run the experiment

Python 3.9+; CPU by default. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m burgers.data --config configs/default.json --output artifacts/data.npz
python -m burgers.train --data artifacts/data.npz --output artifacts/run --model both
python -m burgers.evaluate --data artifacts/data.npz --run artifacts/run
```

Data generation first checks independent initial conditions against a doubled grid
and a halved CFL number; it stops if either maximum relative L2 difference exceeds
0.1%. Training uses only the training and validation splits; evaluation requires
both checkpoints. Tests include an analytic Cole–Hopf solution, conservation,
energy decay, nonlinear behavior, convergence, and a small end-to-end experiment.
GitHub Actions runs these tests without running the full training benchmark.

For the walkthrough, install `jupyter` separately and open `FNO_Burgers.ipynb`.
Run the notebook from this repository's root directory.

## Experiment design

- **Initial conditions:** sums of eight sine/cosine modes with random Gaussian
  coefficients decaying as mode number^(-1.5), rescaled to RMS amplitudes 0.2–0.6.
  They have zero spatial mean. Fourier coefficients are retained so the same
  continuous functions can be evaluated on different grids.
- **Physics:** viscosity 0.02, final time 0.5, periodic domain of length 2. These
  differ from the original notebook deliberately: this first experiment uses a
  fixed, well-resolved viscous regime. Other viscosities and times are not inputs
  to the model and require separate training experiments.
- **Targets:** solve at 512 points, then sample onto 256 points. Periodic grids
  exclude the duplicated endpoint. The solver uses exact Fourier diffusion
  substeps, RK4 conservative transport, 2/3 dealiasing, and Strang splitting
  (second order in time). An adaptive advective CFL limit sets the time step.
- **Split:** seeded, disjoint 700/150/150 training/validation/test cases. Solver
  verification cases use a separate seed. Test cases are never used to select a
  checkpoint or learning schedule.
- **FNO:** lift one field into 32 feature channels, four spectral blocks with
  learned complex weights per mode and a pointwise branch, spatial GELU
  activations, and projection back to one field. A residual from the initial
  field is used in both learned models. No absolute position channel is needed
  for this translation-invariant periodic problem.
- **CNN:** four circular convolution layers of width 32. Real scalar parameter counts (complex weights count twice) are
  reported; models are not parameter-matched. Both use the same data, epochs,
  optimizer, schedule, and relative L2 objective.
- **Training:** 100 epochs, batches of 32, Adam, cosine learning-rate schedule,
  best validation checkpoint. Seeds and deterministic CPU algorithms are set;
  exact equality across different hardware/library versions is not guaranteed.

## Results and interpretation

See [the measured reference run](results/reference/report.md) for actual results,
including median and worst-case plots. `metrics.json` contains the complete
configuration, hardware details, accuracy, timings, physical diagnostics, and
solver convergence checks. Training histories record library versions and costs.
These are measurements from one seed, not a claim of universal performance.

The reference run achieved **0.47% mean relative L2 error** on 150 held-out cases
(worst: 4.10%), versus 28.06% for the CNN and 40.81% for heat diffusion. On the
same 16 held-out waves, FNO error remained about 0.49% at 256, 512, and 1,024
points without retraining. None of the 150 FNO predictions increased energy,
but maximum spatial mean drift was 0.00183; conservation is not exact.

**A neural model is not automatically the fastest useful solver here.** On the
16-case benchmark a 64-point numerical solve had 0.0306% mean error and took
about 6.5 ms, versus about 23 ms and 0.49% error for FNO inference at 256 points.
This experiment demonstrates learning and grid transfer; it does not establish
an accuracy-matched speed advantage over a tuned coarse numerical solver.

Evaluation reports mean, median, 95th percentile, and worst relative L2 errors,
MSE and MAE. It diagnoses spatial mean drift and energy increases. Mean and
energy behavior are not enforced by the network, so failures remain visible.

Resolution transfer evaluates the same 16 held-out functions at 256, 512, and
1,024 points against targets computed on grids at least twice as fine. The grid and time
convergence of these targets are checked separately. This tests grid transfer,
not extrapolation to different physics or initial-condition distributions.

Neural inference is timed on the CPU after warm-up over five repetitions. The
reference solver is timed on the same waves and CPU on its finer grid; reported
speedups are **against that implementation and grid**, not an accuracy-matched
comparison against the fastest available numerical method. A coarse-solver accuracy/runtime curve at 32, 64, 128, and 256 points is included
for comparison, with Fourier interpolation and timing on the same held-out waves.
Data-generation and training costs are reported separately. Neural predictions are approximate;
the reference solver has a much tighter numerical convergence threshold.

Useful future experiments: multiple training seeds, broader or shifted initial
conditions, parameter-matched baselines, viscosity/time conditioning, and
broader solver/runtime benchmarks. Start by editing
`configs/default.json`, generating a new dataset, and using a new run directory.
Convergence checks cover sampled cases and do not prove accuracy for every
possible field; verify refinement again when changing the physical regime.

## Files

| File | Purpose |
|---|---|
| `burgers/solver.py` | Periodic nonlinear reference solver and heat baseline |
| `burgers/data.py` | Seeded dataset, fixed splits, and convergence gate |
| `burgers/models.py` | Standard FNO and periodic CNN |
| `burgers/train.py` | Validation-selected training and checkpoints |
| `burgers/evaluate.py` | Test metrics, physics checks, timing, plots, grid transfer |
| `configs/default.json` | Complete reference experiment configuration |
| `FNO_Burgers.ipynb` | Walkthrough calling the reproducible pipeline |
| `tests/` | Numerical correctness and end-to-end checks |
| `results/reference/` | Recorded measurements and plots; no generated dataset |

## References

- Li et al., [Fourier Neural Operator for Parametric Partial Differential
  Equations](https://arxiv.org/abs/2010.08895), ICLR 2021.
- [NeuralOperator's FNO theory guide](https://neuraloperator.github.io/dev/theory_guide/fno.html).
