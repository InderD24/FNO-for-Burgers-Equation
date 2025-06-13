# FNO for Burgers' Equation

This repository demonstrates a minimal example of solving the one‑dimensional Burgers' equation using a **Fourier Neural Operator (FNO)** implemented in PyTorch. The code lives in a single notebook [`FNO_Burgers.ipynb`](FNO_Burgers.ipynb) which generates training data, defines the FNO model, trains it and evaluates its performance.

## Getting Started

1. Install dependencies (tested with Python 3.9):
   ```bash
   pip install numpy torch matplotlib scikit-learn
   ```
2. Open the notebook and run all cells:
   ```bash
   jupyter notebook FNO_Burgers.ipynb
   ```
   The notebook will:
   - Generate synthetic data for the Burgers' equation.
   - Train a two layer FNO with a mean squared error loss.
   - Output metrics such as MSE, RMSE, MAE and $R^2$.

## Example Output
A typical training run prints progress every 10 epochs and finishes in under a second on a CPU:

```
Epoch 0, Loss: 0.5046
...
Epoch 90, Loss: 0.001386
Training completed in: 0.66 seconds
Test Loss: 0.000548
MSE: 0.000548
RMSE : 0.0234
MAE: 0.0179
R^2: 0.9989
```

## Repository Structure

- `FNO_Burgers.ipynb` – main notebook containing data generation, model definition and training code.
- `README.md` – project overview (this file).

## References

This example follows ideas from the original Fourier Neural Operator paper:
> Zongyi Li, Nikos Kovachki, Kamyar Azizzadenesheli, Burigede Liu, et al. *Fourier Neural Operator for Parametric Partial Differential Equations*. ICLR 2021.

Feel free to experiment with the notebook parameters such as number of modes, width and training epochs
