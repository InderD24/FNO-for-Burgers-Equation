"""Standard 1D FNO blocks and a periodic convolutional baseline."""
import torch
from torch import nn


class SpectralConv1d(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.modes = modes
        self.weight = nn.Parameter(torch.randn(width, width, modes, dtype=torch.cfloat) / width)

    def forward(self, x):
        spectrum = torch.fft.rfft(x)
        count = min(self.modes, spectrum.shape[-1])
        out = torch.zeros_like(spectrum)
        out[..., :count] = torch.einsum('bik,iok->bok', spectrum[..., :count], self.weight[..., :count])
        return torch.fft.irfft(out, n=x.shape[-1])


class FNO1d(nn.Module):
    def __init__(self, modes=16, width=32, layers=4):
        super().__init__()
        self.lift = nn.Conv1d(1, width, 1)
        self.spectral = nn.ModuleList([SpectralConv1d(width, modes) for _ in range(layers)])
        self.local = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(layers)])
        self.project = nn.Sequential(nn.Conv1d(width, 2*width, 1), nn.GELU(), nn.Conv1d(2*width, 1, 1))

    def forward(self, x):
        v = self.lift(x)
        for spectral, local in zip(self.spectral, self.local):
            v = torch.nn.functional.gelu(spectral(v) + local(v))
        return self.project(v) + x


class PeriodicCNN(nn.Module):
    def __init__(self, width=32, layers=4):
        super().__init__()
        blocks = []
        channels = 1
        for _ in range(layers):
            blocks.extend([nn.Conv1d(channels, width, 5, padding=2, padding_mode='circular'), nn.GELU()])
            channels = width
        blocks.append(nn.Conv1d(width, 1, 1))
        self.network = nn.Sequential(*blocks)

    def forward(self, x):
        return self.network(x) + x


def build_model(name, config):
    if name == 'fno':
        return FNO1d(config['modes'], config['width'], config['layers'])
    if name == 'cnn':
        return PeriodicCNN(config['width'], config['layers'])
    raise ValueError(f'Unknown model: {name}')
