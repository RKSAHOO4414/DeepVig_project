# profile_vig.py
"""
Profile Pyramid ViG-Ti to measure latency breakdown:
  - Graph construction (kNN)
  - Graph convolution
  - FFN
  - Total forward pass
  - GPU memory usage

Uses torch.profiler with CUDA activities.
"""

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function
from vig_ti_model import pyramid_vig_ti


def profile_model(model, input_size=(1, 3, 224, 224), device='cuda', warmup=10, runs=50):
    """
    Profile the model and return per-component timings.
    """
    model.eval().to(device)
    dummy = torch.randn(*input_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)

    torch.cuda.synchronize()

    # --- Full forward pass profiling ---
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        with torch.no_grad():
            for _ in range(runs):
                _ = model(dummy)

    print('\n' + '=' * 80)
    print('PROFILING RESULTS: Pyramid ViG-Ti (224x224, batch=1)')
    print('=' * 80)

    # Print top-20 ops by CUDA time
    print('\n--- Top 20 CUDA operations ---')
    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=20))

    # --- Component-level profiling using record_function ---
    print('\n--- Component-level timing (manually instrumented) ---')
    # We instrument the Grapher module by monkey-patching forward
    from vig_ti_model import Grapher

    timings = {'knn': [], 'graph_conv': [], 'ffn': [], 'total': []}

    original_grapher_forward = Grapher.forward

    def instrumented_forward(self, x):
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t2 = torch.cuda.Event(enable_timing=True)
        t3 = torch.cuda.Event(enable_timing=True)
        t4 = torch.cuda.Event(enable_timing=True)

        t0.record()
        B, C, H, W = x.shape
        edge_index, B, N = self._build_knn_graph(x)
        t1.record()

        x_flat = x.permute(0, 2, 3, 1).reshape(B * N, C)
        x_conv = self.graph_conv(x_flat, edge_index)
        x_conv = x_conv.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        t2.record()

        x_ffn = self.ffn(x_conv)
        t3.record()

        x = self.drop_path(x_ffn) + x
        t4.record()

        torch.cuda.synchronize()
        timings['knn'].append(t0.elapsed_time(t1))
        timings['graph_conv'].append(t1.elapsed_time(t2))
        timings['ffn'].append(t2.elapsed_time(t3))
        timings['total'].append(t0.elapsed_time(t4))
        return x

    Grapher.forward = instrumented_forward

    with torch.no_grad():
        for _ in range(runs):
            _ = model(dummy)
    torch.cuda.synchronize()

    Grapher.forward = original_grapher_forward

    # Report averages
    import numpy as np
    print('\n--- Average per-Grapher-layer timings (ms) ---')
    for k in ['knn', 'graph_conv', 'ffn', 'total']:
        arr = np.array(timings[k])
        print(f'  {k:12s}: {arr.mean():.3f} ms  (std: {arr.std():.3f})')

    # Memory
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(dummy)
    print(f'\nPeak GPU memory: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB')

    return prof


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Profiling on: {device}')

    model = pyramid_vig_ti(num_classes=1000)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Pyramid ViG-Ti: {n_params / 1e6:.2f}M parameters')

    profile_model(model, input_size=(1, 3, 224, 224), device=device)

    # Also profile with larger batch
    print('\n' + '=' * 80)
    print('PROFILING WITH BATCH SIZE 32')
    print('=' * 80)
    profile_model(model, input_size=(32, 3, 224, 224), device=device)


if __name__ == '__main__':
    main()