# phase0_setup.py
"""
Phase 0 – Setup: Main entry point.

Runs both Phase 0 experiments:
  0.1  Reproduce Pyramid ViG-Ti baseline on ImageNet-100
  0.2  Profile ViG-Ti (graph construction vs. conv vs. FFN)
"""

import argparse
import subprocess
import sys
import os


def run_experiment_0_1(args):
    """Experiment 0.1: Baseline reproduction on ImageNet-100."""
    print('\n' + '=' * 80)
    print('EXPERIMENT 0.1: Reproduce Pyramid ViG-Ti on ImageNet-100')
    print('=' * 80)

    cmd = [
        sys.executable, 'train_imagenet100.py',
        '--data_dir', args.data_dir,
        '--output_dir', args.output_dir,
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--weight_decay', str(args.weight_decay),
        '--warmup_epochs', str(args.warmup_epochs),
        '--num_workers', str(args.num_workers),
    ]
    subprocess.run(cmd, check=True)


def run_experiment_0_2(args):
    """Experiment 0.2: Profile ViG-Ti."""
    print('\n' + '=' * 80)
    print('EXPERIMENT 0.2: Profile ViG-Ti (no training)')
    print('=' * 80)

    cmd = [sys.executable, 'profile_vig.py']
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description='Phase 0 – Setup')
    parser.add_argument('--data_dir', type=str, default='./data/imagenet100')
    parser.add_argument('--output_dir', type=str, default='./outputs/vig_ti_imagenet100')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--skip_train', action='store_true', help='Skip Experiment 0.1')
    parser.add_argument('--skip_profile', action='store_true', help='Skip Experiment 0.2')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print('=' * 80)
    print('PHASE 0 – SETUP')
    print('=' * 80)
    print(f'  Experiment 0.1: Baseline reproduction ({"SKIP" if args.skip_train else "RUN"})')
    print(f'  Experiment 0.2: Profiling ({"SKIP" if args.skip_profile else "RUN"})')
    print('=' * 80)

    if not args.skip_train:
        run_experiment_0_1(args)
    if not args.skip_profile:
        run_experiment_0_2(args)

    print('\n' + '=' * 80)
    print('PHASE 0 COMPLETE')
    print('=' * 80)
    print(f'Outputs saved to: {args.output_dir}')
    print('Next: Phase 1 – Pilot Experiments (ImageNet-100 depth scaling & anti-smoothing)')


if __name__ == '__main__':
    main()