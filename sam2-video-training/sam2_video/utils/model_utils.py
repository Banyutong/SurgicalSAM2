"""
Model utility functions: parameter counting, trainable setup, and config io.
"""

from typing import Any, Dict, List
from torch import nn
from loguru import logger
import yaml


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_model_info(
    model: nn.Module,
    checkpoint_path: str,
    config_path: str,
    device: str,
) -> Dict[str, Any]:
    trainable_params = count_trainable_parameters(model)
    total_params = count_total_parameters(model)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_ratio": trainable_params / total_params if total_params > 0 else 0,
        "checkpoint_path": checkpoint_path,
        "config_path": config_path,
        "device": device,
    }


def setup_trainable_modules(
    model: nn.Module, module_mapping: Dict[str, nn.Module], trainable_modules: List[str]
) -> None:
    all_modules = list(module_mapping.keys())
    for module_name in all_modules:
        module = module_mapping.get(module_name)
        if module is None:
            continue
        is_trainable = module_name in trainable_modules
        for param in module.parameters():
            param.requires_grad = is_trainable
        logger.info(f"Module '{module_name}': {'trainable' if is_trainable else 'frozen'}")


def freeze_module_by_name(module_mapping: Dict[str, nn.Module], module_name: str) -> None:
    module = module_mapping.get(module_name)
    if module is None:
        raise KeyError(f"Module '{module_name}' not found")
    for param in module.parameters():
        param.requires_grad = False
    logger.info(f"Module '{module_name}' frozen")


def unfreeze_module_by_name(module_mapping: Dict[str, nn.Module], module_name: str) -> None:
    module = module_mapping.get(module_name)
    if module is None:
        raise KeyError(f"Module '{module_name}' not found")
    for param in module.parameters():
        param.requires_grad = True
    logger.info(f"Module '{module_name}' unfrozen")


def get_trainable_module_names(module_mapping: Dict[str, nn.Module]) -> List[str]:
    return [name for name, m in module_mapping.items() if m is not None and any(p.requires_grad for p in m.parameters())]


def save_model_config(config_dict: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

