"""Train on the training split; select checkpoint using validation only."""
import argparse
import json
import random
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from .models import build_model


def seed_everything(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def relative_l2(pred, target):
    numerator = torch.linalg.vector_norm((pred-target).flatten(1), dim=1)
    denominator = torch.linalg.vector_norm(target.flatten(1), dim=1).clamp_min(1e-8)
    return (numerator / denominator).mean()


def fit(data_path, output_dir, name='fno'):
    data_path, output_dir = Path(data_path), Path(output_dir)
    metadata = json.loads(data_path.with_suffix('.json').read_text())
    config = metadata['config']
    seed_everything(config['seed'], config['threads'])
    with np.load(data_path) as data:
        x = torch.from_numpy(data['x'].copy()).unsqueeze(1)
        y = torch.from_numpy(data['y'].copy()).unsqueeze(1)
        train, val = data['train'], data['validation']
    generator = torch.Generator().manual_seed(config['seed'])
    loader = DataLoader(TensorDataset(x[train], y[train]), batch_size=config['batch_size'], shuffle=True, generator=generator)
    validation_loader = DataLoader(TensorDataset(x[val], y[val]), batch_size=config['batch_size'])
    model = build_model(name, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    best, history = float('inf'), []
    output_dir.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        for inputs, targets in loader:
            optimizer.zero_grad()
            loss = relative_l2(model(inputs), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(inputs)
        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for inputs, targets in validation_loader:
                val_loss += relative_l2(model(inputs), targets).item() * len(inputs)
        val_loss /= len(val)
        history.append({'epoch': epoch+1, 'train_relative_l2': train_loss/len(train), 'validation_relative_l2': val_loss})
        if val_loss < best:
            best = val_loss
            torch.save({'model': name, 'state_dict': model.state_dict(), 'config': config,
                        'epoch': epoch+1, 'validation_relative_l2': best}, output_dir/f'{name}.pt')
        scheduler.step()
        if epoch == 0 or (epoch+1) % 10 == 0:
            print(f'{name}: epoch {epoch+1}, validation relative L2 {val_loss:.4f}', flush=True)
    report = {'config':config, 'history':history, 'training_seconds':perf_counter()-start,
              'parameters':sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters()), 'torch_version':str(torch.__version__),
              'numpy_version':np.__version__, 'device':'cpu', 'best_validation_relative_l2':best}
    (output_dir/f'{name}_training.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='artifacts/data.npz')
    parser.add_argument('--output', default='artifacts/run')
    parser.add_argument('--model', choices=['fno', 'cnn', 'both'], default='both')
    args = parser.parse_args()
    for name in (['fno', 'cnn'] if args.model == 'both' else [args.model]):
        fit(args.data, args.output, name)


if __name__ == '__main__':
    main()
