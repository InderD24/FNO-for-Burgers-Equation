"""Dealiased Fourier solver: Strang diffusion splitting + RK4 transport.

Domain [-1, 1), periodic boundaries; u_t + (u^2/2)_x = nu u_xx.
Diffusion is integrated exactly in Fourier space. The nonlinear substep
uses conservative derivatives and the 2/3 truncation rule. Splitting is
second order in time; convergence must be checked for each regime.
"""
import numpy as np


def grid(n):
    return np.linspace(-1.0, 1.0, n, endpoint=False)


def heat_solution(u0, viscosity, time):
    k = np.pi * np.arange(u0.shape[-1] // 2 + 1)
    return np.fft.irfft(np.fft.rfft(u0, axis=-1) *
                        np.exp(-viscosity * k**2 * time), n=u0.shape[-1], axis=-1)


def solve(u0, viscosity=0.02, time=0.5, cfl=0.2):
    u = np.asarray(u0, dtype=np.float64).copy()
    n = u.shape[-1]
    if n < 16 or n % 2 or viscosity <= 0 or time < 0 or not 0 < cfl <= 0.4:
        raise ValueError('Use an even grid >=16, positive viscosity, nonnegative time, and 0<cfl<=0.4')
    if not np.isfinite(u).all():
        raise ValueError('Initial condition must be finite')
    if time == 0:
        return u
    k = np.pi * np.arange(n // 2 + 1)
    keep = np.arange(n // 2 + 1) < n / 3
    u = np.fft.irfft(np.fft.rfft(u, axis=-1) * keep, n=n, axis=-1)

    def rhs(v):
        flux = np.fft.rfft(0.5 * v**2, axis=-1) * keep
        return np.fft.irfft(-1j * k * flux, n=n, axis=-1)

    elapsed = 0.0
    while elapsed < time:
        dt = min(cfl * (2.0 / n) / max(float(np.max(np.abs(u))), 0.1), time - elapsed)
        decay = np.exp(-viscosity * k**2 * dt / 2)
        v = np.fft.irfft(np.fft.rfft(u, axis=-1) * decay, n=n, axis=-1)
        a = rhs(v)
        b = rhs(v + dt * a / 2)
        c = rhs(v + dt * b / 2)
        d = rhs(v + dt * c)
        v = v + dt * (a + 2*b + 2*c + d) / 6
        u = np.fft.irfft(np.fft.rfft(v, axis=-1) * decay * keep, n=n, axis=-1)
        if not np.isfinite(u).all():
            raise FloatingPointError('Solver diverged; reduce CFL or refine the grid')
        elapsed += dt
    return u


def upsample_periodic(u, n):
    """Fourier interpolation from an even periodic grid to a finer even grid."""
    old_n = u.shape[-1]
    if n < old_n or n % 2 or old_n % 2:
        raise ValueError('Use even grids with output size >= input size')
    if n == old_n:
        return np.asarray(u).copy()
    spectrum = np.fft.rfft(u, axis=-1)
    spectrum[..., -1] *= 0.5  # split the old Nyquist mode into +/- frequencies
    return np.fft.irfft(spectrum, n=n, axis=-1) * (n / old_n)
