#!/usr/bin/env python3
"""
Shared device selection.

`torch.cuda.is_available()` returns True whenever a CUDA driver and device are
visible, which on a cluster login node (or a compute node without --gres=gpu)
they often are even though no GPU is actually allocated to us. The failure then
shows up much later, at the first .to(device) or torch.load(map_location=...),
as an opaque "CUDA-capable device(s) is/are busy or unavailable".

pick_device() probes the device with a real allocation and falls back to CPU, so
scripts run wherever they are launched instead of crashing.
"""
import torch


def pick_device(prefer='auto', quiet=False):
    """
    Return a usable torch.device.

    prefer: 'auto' (cuda if genuinely usable, else cpu), 'cpu', or 'cuda'
            ('cuda' is returned as-is so a real GPU job still fails loudly if
             the GPU is broken, rather than silently running 100x slower on CPU)
    """
    if prefer == 'cpu':
        return torch.device('cpu')
    if prefer == 'cuda':
        return torch.device('cuda')

    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            return torch.device('cuda')
        except RuntimeError:
            if not quiet:
                print('CUDA visible but not allocated (no --gres=gpu?); using CPU.')
    return torch.device('cpu')
