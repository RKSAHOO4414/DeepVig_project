# train_imagenet100.py
"""Train Pyramid ViG-Ti on ImageNet-100 (Experiment 0.1)."""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from timm.scheduler import CosineLRScheduler
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

from vig_ti_model import pyramid_vig_ti
from utils.data import build_imagenet100_loaders
from utils.logger import WandbLogger


def train_one_epoch(model, loader, optimizer, scaler, criterion, mixup_fn, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        _, pred = outputs.max(1)
        if mixup_fn is None:
            correct += pred.eq(targets).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * images.size(0)
        _, pred = outputs.max(1)
        correct += pred.eq(targets).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Data
    train_loader, val_loader, num_classes = build_imagenet100_loaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )
    print(f'ImageNet-100: {num_classes} classes, {len(train_loader.dataset)} train images')

    # Model
    model = pyramid_vig_ti(num_classes=num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Pyramid ViG-Ti: {n_params / 1e6:.2f}M parameters')

    # Mixup / CutMix
    mixup_fn = Mixup(
        mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
        prob=1.0, switch_prob=0.5, mode='batch',
        label_smoothing=0.1, num_classes=num_classes,
    )
    train_criterion = SoftTargetCrossEntropy()
    val_criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler()

    # LR scheduler
    scheduler = CosineLRScheduler(
        optimizer, t_initial=args.epochs, lr_min=1e-6,
        warmup_t=args.warmup_epochs, warmup_lr_init=1e-6,
    )

    # Logger
    logger = WandbLogger(project='DeepViG-Phase0', name='ViG-Ti-ImageNet100')

    best_acc = 0.0
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, train_criterion, mixup_fn, device
        )
        val_loss, val_acc = validate(model, val_loader, val_criterion, device)
        scheduler.step(epoch)

        logger.log({
            'epoch': epoch,
            'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr'],
            'time': time.time() - t0,
        })

        print(f'Epoch {epoch+1}/{args.epochs} | '
              f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | '
              f'Time: {time.time()-t0:.1f}s')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best.pth'))
            print(f'  -> New best accuracy: {best_acc:.4f}')

    print(f'\nBaseline reproduction complete. Best validation accuracy: {best_acc:.4f}')
    logger.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/imagenet100')
    parser.add_argument('--output_dir', type=str, default='./outputs/vig_ti_imagenet100')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)