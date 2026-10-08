import unittest
import numpy as np
from burgers.solver import grid, solve, heat_solution, upsample_periodic
from burgers.data import sample_coefficients, initial_conditions, verify_solver


class SolverTests(unittest.TestCase):
    def test_constant_and_zero_time(self):
        u = np.full((2, 64), 0.4)
        np.testing.assert_allclose(solve(u), u, atol=1e-12)
        np.testing.assert_array_equal(solve(u, time=0), u)

    def test_heat_mode_exact_decay(self):
        u = np.sin(np.pi*grid(64))
        np.testing.assert_allclose(heat_solution(u, 0.02, 0.5), np.exp(-0.02*np.pi**2*0.5)*u, atol=1e-12)

    def test_nonlinearity_mean_and_energy(self):
        u = 0.6*np.sin(np.pi*grid(128)) + 0.2
        result = solve(u)
        self.assertGreater(np.linalg.norm(result-heat_solution(u, 0.02, 0.5)), 0.5)
        self.assertAlmostEqual(float(result.mean()), float(u.mean()), places=12)
        self.assertLess(np.mean(result**2), np.mean(u**2))

    def test_manufactured_cole_hopf_solution(self):
        # phi=1+a exp(-nu*pi^2*t) cos(pi*x); u=-2nu phi_x/phi.
        # Independent analytic nonlinear solution validates signs and diffusion.
        nu, a, time = 0.1, 0.5, 0.3
        x = grid(128)
        def exact(t):
            b = a*np.exp(-nu*np.pi**2*t)
            return 2*nu*b*np.pi*np.sin(np.pi*x)/(1+b*np.cos(np.pi*x))
        np.testing.assert_allclose(solve(exact(0), nu, time, cfl=0.1), exact(time), atol=2e-6)

    def test_convergence(self):
        coeff = sample_coefficients(3, 6, 5)
        report = verify_solver(coeff, 256, 0.02, 0.5, 0.2, 0.001)
        self.assertTrue(report['passed'])

    def test_same_function_at_multiple_resolutions(self):
        coeff = sample_coefficients(3, 6, 42)
        np.testing.assert_allclose(initial_conditions(coeff, 128), initial_conditions(coeff, 512)[:, ::4], atol=1e-12)

    def test_fourier_interpolation(self):
        x = grid(32)
        u = np.sin(3*np.pi*x) + 0.2*np.cos(16*np.pi*x)
        expected = np.sin(3*np.pi*grid(128)) + 0.2*np.cos(16*np.pi*grid(128))
        np.testing.assert_allclose(upsample_periodic(u, 128), expected, atol=1e-12)

    def test_invalid_parameters(self):
        for kwargs in [{'viscosity':0}, {'time':-1}, {'cfl':1}]:
            with self.assertRaises(ValueError):
                solve(np.zeros(64), **kwargs)


if __name__ == '__main__':
    unittest.main()
