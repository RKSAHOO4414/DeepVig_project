# utils/checkpoint.py
"""Checkpoint save/load utilities."""

import os
import torch


def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f'Checkpoint saved: {path}')


def load_checkpoint(model, optimizer, path, device='cuda'):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    print(f'Checkpoint loaded from epoch {ckpt["epoch"]}')
    return ckpt['epoch']