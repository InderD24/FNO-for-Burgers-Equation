import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from burgers.data import generate
from burgers.models import FNO1d
from burgers.train import fit
from burgers.evaluate import evaluate


class PipelineTests(unittest.TestCase):
    def test_resolution_and_gradients(self):
        model = FNO1d(modes=8, width=8, layers=2)
        for n in [64, 128, 256]:
            x = torch.randn(2, 1, n)
            pred = model(x)
            self.assertEqual(pred.shape, x.shape)
            pred.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_reproducible_splits_checkpoint_and_evaluation(self):
        config = json.loads(Path('configs/default.json').read_text())
        config.update(samples=20, grid_points=32, reference_points=64, initial_modes=3,
                      time=0.1, epochs=2, width=8, layers=2, modes=6, batch_size=8,
                      evaluation_resolutions=[32, 64], resolution_samples=2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ['a', 'b']:
                generate(config, root/f'{name}.npz')
            with np.load(root/'a.npz') as a, np.load(root/'b.npz') as b:
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])
                splits = [set(a[k]) for k in ['train', 'validation', 'test']]
                self.assertEqual(len(set.union(*splits)), config['samples'])
                self.assertTrue(all(not splits[i] & splits[j] for i in range(3) for j in range(i)))
            for name in ['fno', 'cnn']:
                fit(root/'a.npz', root/'run', name)
            report = evaluate(root/'a.npz', root/'run')
            self.assertEqual(set(report['test']), {'fno', 'cnn', 'identity', 'heat'})
            self.assertEqual(len(report['resolution_transfer']), 2)
            self.assertTrue((root/'run'/'predictions.png').is_file())


if __name__ == '__main__':
    unittest.main()
