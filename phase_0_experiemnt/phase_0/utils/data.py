# utils/data.py
"""Dataset loaders and transforms for ImageNet-100."""

import os
from torchvision import datasets, transforms
from timm.data import Mixup, create_transform


def build_imagenet100_loaders(data_dir, batch_size=128, num_workers=8):
    """
    Build train/val loaders for ImageNet-100.

    Expected directory structure:
        data_dir/
            train/
                n01440764/  ... (100 class folders)
            val/
                n01440764/  ... (100 class folders)
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    # DeiT-style training transforms
    train_transform = create_transform(
        input_size=224,
        is_training=True,
        color_jitter=0.4,
        auto_augment='rand-m9-mstd0.5-inc1',
        interpolation='bicubic',
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
    )

    val_transform = create_transform(
        input_size=224,
        is_training=False,
        interpolation='bicubic',
        crop_pct=0.95,
    )

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, len(train_dataset.classes)