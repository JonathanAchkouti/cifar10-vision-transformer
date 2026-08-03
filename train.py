import argparse
import csv
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

from data import get_loaders
from model import build_model, count_parameters


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def build_optimizer(model, h):
    """SGD for training from scratch, AdamW for fine-tuning.

    AdamW adapts a per-parameter step size, which suits fine-tuning: the
    pretrained layers need small careful updates while the freshly
    initialised classification head needs larger ones. SGD applies one global
    learning rate and would either move the head too slowly or damage the
    backbone.
    """
    name = h.get('optimizer', 'sgd').lower()
    if name == 'adamw':
        return optim.AdamW(model.parameters(), lr=h['max_lr'],
                           weight_decay=h['weight_decay'])
    return optim.SGD(model.parameters(), lr=h['max_lr'],
                     momentum=h['momentum'], nesterov=True,
                     weight_decay=h['weight_decay'])


@torch.no_grad()
def evaluate(model, loader, criterion, device, tta=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        if tta:
            outputs = (outputs + model(torch.flip(images, dims=[3]))) / 2
        total_loss += criterion(outputs, labels).item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100 * correct / total


def train(model, train_loader, val_loader, criterion, optimizer,
          scheduler, scaler, device, config):
    epochs = config['hyperparameters']['epochs']
    ckpt_path = config['paths']['model_path']
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    use_amp = device.type == 'cuda'
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss, seen = 0.0, 0
        bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch')

        for inputs, labels in bar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)
            bar.set_postfix({'loss': running_loss / seen,
                             'lr': scheduler.get_last_lr()[0]})

        train_loss = running_loss / seen
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        marker = ''
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            marker = '  <- best'

        print(f'Epoch {epoch + 1}: train loss {train_loss:.3f} | '
              f'val loss {val_loss:.3f} | val acc {val_acc:.2f}%{marker}')

    print(f'Training finished. Best validation accuracy: {best_val_acc:.2f}%')
    return best_val_acc


def log_result(config, params, val_acc, test_acc, minutes):
    path = config['paths']['results_path']
    h, d, m = config['hyperparameters'], config['data'], config['model']
    row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'run_name': config['run_name'],
        'model': m['name'],
        'pretrained': m.get('pretrained', False),
        'params': params,
        'epochs': h['epochs'],
        'batch_size': h['batch_size'],
        'optimizer': h.get('optimizer', 'sgd'),
        'max_lr': h['max_lr'],
        'augment': d['augment'],
        'cutout': d['cutout'],
        'label_smoothing': h['label_smoothing'],
        'tta': config['eval']['tta'],
        'val_acc': round(val_acc, 2),
        'test_acc': round(test_acc, 2),
        'minutes': round(minutes, 1),
    }
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f'Logged to {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(config['hyperparameters']['seed'])
    device = get_device()
    print(f"Run '{config['run_name']}' on {device}")

    # The model is built first because it dictates the input resolution and
    # normalization statistics the data pipeline must use.
    model, mean, std, image_size = build_model(config)
    model = model.to(device)
    print(f"Model: {config['model']['name']} "
          f'({count_parameters(model):,} trainable parameters)')

    train_loader, val_loader, test_loader = get_loaders(
        config, device, mean, std, image_size)

    h = config['hyperparameters']
    criterion = nn.CrossEntropyLoss(label_smoothing=h['label_smoothing'])
    optimizer = build_optimizer(model, h)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=h['max_lr'],
        steps_per_epoch=len(train_loader), epochs=h['epochs'],
        pct_start=h.get('pct_start', 0.25),
        div_factor=10, final_div_factor=100,
    )
    scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')

    start = time.time()
    val_acc = train(model, train_loader, val_loader, criterion,
                    optimizer, scheduler, scaler, device, config)
    minutes = (time.time() - start) / 60

    model.load_state_dict(torch.load(config['paths']['model_path'],
                                     map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device,
                                   tta=config['eval']['tta'])
    print(f'\nTEST: loss {test_loss:.3f} | accuracy {test_acc:.2f}% '
          f'| {minutes:.1f} min')

    log_result(config, count_parameters(model), val_acc, test_acc, minutes)


if __name__ == '__main__':
    main()
