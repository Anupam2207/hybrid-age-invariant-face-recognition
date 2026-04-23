"""Runtime and hardware-aware defaults for training and inference.

The target machine for this project is a Windows desktop with an NVIDIA GTX 1050
Ti (4 GB VRAM) and 8 GB system RAM. These helpers keep batch sizes, image sizes,
and data-loader settings realistic for that hardware while still allowing manual
overrides from the command line.
"""

from __future__ import annotations

import os
import platform
from dataclasses import asdict, dataclass
from typing import Any, Dict

import torch


@dataclass
class RuntimeProfile:
    profile_name: str
    device: str
    device_name: str
    total_vram_gb: float
    backbone: str
    image_size: int
    batch_size: int
    eval_batch_size: int
    cache_batch_size: int
    gradient_accumulation_steps: int
    num_workers: int
    amp: bool
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int | None
    checkpoint_frequency: int
    freeze_backbone_epochs: int
    effective_batch_size: int
    windows_low_memory_mode: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



def resolve_device(device_name: str) -> torch.device:
    requested = str(device_name).strip().lower()
    if requested == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(device_name)



def describe_device(device: torch.device) -> tuple[str, float]:
    if device.type == 'cuda' and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        return props.name, float(props.total_memory) / (1024.0 ** 3)
    if device.type == 'mps':
        return 'Apple Metal (MPS)', 0.0

    cpu_name = platform.processor() or os.environ.get('PROCESSOR_IDENTIFIER', 'CPU')
    return cpu_name, 0.0



def _auto_workers() -> int:
    cpu_count = os.cpu_count() or 1
    is_windows = platform.system().lower().startswith('win')
    if is_windows:
        # Windows worker processes are significantly heavier. On an 8 GB machine,
        # 0 workers is the safest default and avoids RAM spikes and multiprocessing
        # friction during development.
        return 0
    return min(2, max(0, cpu_count // 4))



def _defaults_for_device(device: torch.device, requested_backbone: str) -> dict[str, Any]:
    device_name, total_vram_gb = describe_device(device)
    is_windows = platform.system().lower().startswith('win')
    requested_backbone = str(requested_backbone).lower()

    if device.type == 'cuda' and total_vram_gb <= 4.5:
        defaults = {
            'profile_name': 'cuda_4gb_safe',
            'backbone': 'resnet18' if requested_backbone == 'auto' else requested_backbone,
            'image_size': 192,
            'batch_size': 8,
            'eval_batch_size': 16,
            'cache_batch_size': 16,
            'gradient_accumulation_steps': 2,
            'num_workers': 0 if is_windows else 2,
            'amp': True,
            'pin_memory': True,
            'persistent_workers': False,
            'prefetch_factor': 2,
            'checkpoint_frequency': 5,
            'freeze_backbone_epochs': 2,
        }
    elif device.type == 'cuda' and total_vram_gb <= 8.5:
        defaults = {
            'profile_name': 'cuda_midrange',
            'backbone': 'resnet50' if requested_backbone == 'auto' else requested_backbone,
            'image_size': 224,
            'batch_size': 16,
            'eval_batch_size': 32,
            'cache_batch_size': 32,
            'gradient_accumulation_steps': 1,
            'num_workers': 2 if is_windows else 4,
            'amp': True,
            'pin_memory': True,
            'persistent_workers': False,
            'prefetch_factor': 2,
            'checkpoint_frequency': 5,
            'freeze_backbone_epochs': 1,
        }
    elif device.type == 'mps':
        defaults = {
            'profile_name': 'apple_mps_safe',
            'backbone': 'resnet18' if requested_backbone == 'auto' else requested_backbone,
            'image_size': 192,
            'batch_size': 4,
            'eval_batch_size': 8,
            'cache_batch_size': 8,
            'gradient_accumulation_steps': 2,
            'num_workers': 0,
            'amp': False,
            'pin_memory': False,
            'persistent_workers': False,
            'prefetch_factor': None,
            'checkpoint_frequency': 5,
            'freeze_backbone_epochs': 1,
        }
    else:
        defaults = {
            'profile_name': 'cpu_safe',
            'backbone': 'mobilenet_v2' if requested_backbone == 'auto' else requested_backbone,
            'image_size': 160,
            'batch_size': 4,
            'eval_batch_size': 8,
            'cache_batch_size': 8,
            'gradient_accumulation_steps': 1,
            'num_workers': 0,
            'amp': False,
            'pin_memory': False,
            'persistent_workers': False,
            'prefetch_factor': None,
            'checkpoint_frequency': 5,
            'freeze_backbone_epochs': 0,
        }

    # ResNet50 is still supported, but on a 4 GB GPU it needs smaller images,
    # smaller micro-batches, and more accumulation.
    if device.type == 'cuda' and total_vram_gb <= 4.5 and defaults['backbone'] == 'resnet50':
        defaults.update(
            {
                'profile_name': 'cuda_4gb_resnet50_conservative',
                'image_size': 160,
                'batch_size': 4,
                'eval_batch_size': 8,
                'cache_batch_size': 8,
                'gradient_accumulation_steps': 4,
                'freeze_backbone_epochs': 2,
            }
        )

    defaults['device_name'] = device_name
    defaults['total_vram_gb'] = total_vram_gb
    defaults['windows_low_memory_mode'] = is_windows and defaults['num_workers'] == 0
    return defaults



def resolve_runtime_profile(
    device: torch.device,
    requested_backbone: str = 'auto',
    image_size: int = 0,
    batch_size: int = 0,
    eval_batch_size: int = 0,
    cache_batch_size: int = 0,
    gradient_accumulation_steps: int = 0,
    num_workers: int = -1,
    amp: bool = True,
    checkpoint_frequency: int = 5,
    freeze_backbone_epochs: int = -1,
) -> RuntimeProfile:
    defaults = _defaults_for_device(device=device, requested_backbone=requested_backbone)

    resolved_backbone = defaults['backbone'] if requested_backbone == 'auto' else requested_backbone
    resolved_image_size = defaults['image_size'] if int(image_size) <= 0 else int(image_size)
    resolved_batch_size = defaults['batch_size'] if int(batch_size) <= 0 else int(batch_size)
    resolved_eval_batch_size = defaults['eval_batch_size'] if int(eval_batch_size) <= 0 else int(eval_batch_size)
    resolved_cache_batch_size = defaults['cache_batch_size'] if int(cache_batch_size) <= 0 else int(cache_batch_size)
    resolved_accum = (
        defaults['gradient_accumulation_steps'] if int(gradient_accumulation_steps) <= 0 else int(gradient_accumulation_steps)
    )
    resolved_workers = _auto_workers() if int(num_workers) < 0 else int(num_workers)
    resolved_checkpoint_frequency = (
        defaults['checkpoint_frequency'] if int(checkpoint_frequency) <= 0 else int(checkpoint_frequency)
    )
    resolved_freeze_epochs = (
        defaults['freeze_backbone_epochs'] if int(freeze_backbone_epochs) < 0 else int(freeze_backbone_epochs)
    )

    persistent_workers = bool(resolved_workers > 0 and defaults['persistent_workers'])
    prefetch_factor = defaults['prefetch_factor'] if resolved_workers > 0 else None

    resolved_amp = bool(amp and device.type == 'cuda')

    profile = RuntimeProfile(
        profile_name=str(defaults['profile_name']),
        device=str(device),
        device_name=str(defaults['device_name']),
        total_vram_gb=float(defaults['total_vram_gb']),
        backbone=str(resolved_backbone),
        image_size=int(resolved_image_size),
        batch_size=int(resolved_batch_size),
        eval_batch_size=int(resolved_eval_batch_size),
        cache_batch_size=int(resolved_cache_batch_size),
        gradient_accumulation_steps=max(1, int(resolved_accum)),
        num_workers=max(0, int(resolved_workers)),
        amp=resolved_amp,
        pin_memory=bool(device.type == 'cuda' and defaults['pin_memory']),
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        checkpoint_frequency=max(1, int(resolved_checkpoint_frequency)),
        freeze_backbone_epochs=max(0, int(resolved_freeze_epochs)),
        effective_batch_size=int(resolved_batch_size) * max(1, int(resolved_accum)),
        windows_low_memory_mode=bool(defaults['windows_low_memory_mode']),
    )
    return profile



def build_dataloader_kwargs(profile: RuntimeProfile) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        'num_workers': int(profile.num_workers),
        'pin_memory': bool(profile.pin_memory),
    }
    if profile.num_workers > 0:
        kwargs['persistent_workers'] = bool(profile.persistent_workers)
        if profile.prefetch_factor is not None:
            kwargs['prefetch_factor'] = int(profile.prefetch_factor)
    return kwargs
