"""Seeded smooth random Fourier initial conditions and verified targets."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from .solver import grid, solve


def initial_conditions(coefficients, n):
    x = grid(n)
    modes = np.arange(1, coefficients.shape[1] + 1)
    angles = np.pi * modes[:, None] * x[None, :]
    return coefficients[..., 0] @ np.sin(angles) + coefficients[..., 1] @ np.cos(angles)


def sample_coefficients(count, modes, seed):
    rng = np.random.default_rng(seed)
    coefficients = rng.normal(size=(count, modes, 2)) / np.arange(1, modes + 1)[None, :, None]**1.5
    rms = np.sqrt(np.sum(coefficients**2, axis=(1, 2)) / 2)
    amplitudes = rng.uniform(0.2, 0.6, count)
    return coefficients * (amplitudes / rms)[:, None, None]


def relative_errors(pred, target):
    return np.linalg.norm(pred-target, axis=-1) / np.maximum(np.linalg.norm(target, axis=-1), 1e-12)


def verify_solver(coefficients, n, viscosity, time, cfl, tolerance):
    coarse = solve(initial_conditions(coefficients, n), viscosity, time, cfl)
    fine = solve(initial_conditions(coefficients, 2*n), viscosity, time, cfl)
    half_step = solve(initial_conditions(coefficients, n), viscosity, time, cfl/2)
    report = {
        'cases': len(coefficients), 'grid_points': n, 'refined_grid_points': 2*n,
        'max_grid_relative_l2': float(relative_errors(coarse, fine[..., ::2]).max()),
        'max_time_relative_l2': float(relative_errors(coarse, half_step).max()),
        'tolerance': tolerance,
    }
    report['passed'] = max(report['max_grid_relative_l2'], report['max_time_relative_l2']) < tolerance
    if not report['passed']:
        raise RuntimeError(f'Solver convergence check failed: {report}')
    return report


def generate(config, destination):
    count, n = config['samples'], config['grid_points']
    reference_n = config['reference_points']
    if reference_n % n or reference_n < 2*n or config['initial_modes'] >= n/3:
        raise ValueError('Reference grid must be a multiple >=2x training grid; initial modes < N/3')
    if count < 10 or not 0 < config['train_fraction'] < 1 or not 0 < config['validation_fraction'] < 1-config['train_fraction']:
        raise ValueError('Invalid sample count or split fractions')
    coeff = sample_coefficients(count, config['initial_modes'], config['seed'])
    # Independent cases, never taken from the eventual held-out test split.
    verification_coeff = sample_coefficients(8, config['initial_modes'], config['seed'] + 10000)
    check = verify_solver(verification_coeff, reference_n, config['viscosity'], config['time'], config['cfl'], config['solver_tolerance'])
    start = perf_counter()
    x, y = initial_conditions(coeff, n), []
    for offset in range(0, count, config['solver_batch_size']):
        u0 = initial_conditions(coeff[offset:offset+config['solver_batch_size']], reference_n)
        y.append(solve(u0, config['viscosity'], config['time'], config['cfl'])[:, ::reference_n//n])
    y = np.concatenate(y)
    indices = np.random.default_rng(config['seed']+1).permutation(count)
    train_end = int(count * config['train_fraction'])
    val_end = train_end + int(count * config['validation_fraction'])
    splits = dict(train=indices[:train_end], validation=indices[train_end:val_end], test=indices[val_end:])
    if any(len(v) == 0 for v in splits.values()):
        raise ValueError('Every split must contain examples')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, x=x.astype('float32'), y=y.astype('float32'), coefficients=coeff, **splits)
    metadata = {'config': config, 'solver_verification': check, 'generation_seconds': perf_counter()-start,
                'split_sizes': {k:len(v) for k,v in splits.items()}}
    destination.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.json')
    parser.add_argument('--output', default='artifacts/data.npz')
    args = parser.parse_args()
    print(json.dumps(generate(json.loads(Path(args.config).read_text()), args.output), indent=2))


if __name__ == '__main__':
    main()
