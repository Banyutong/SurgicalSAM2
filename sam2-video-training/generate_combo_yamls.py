#!/usr/bin/env python3
"""
Generate combo YAMLs under configs/combo/ from paths listed in eval_list.md.

For each checkpoint path, create three YAML variants following rules:
1) mem: train [memory_encoder, memory_attention]
2) sfx: based on path suffix
   - no suffix => [mask_decoder]
   - pe        => [mask_decoder, prompt_encoder]
   - all       => [mask_decoder, prompt_encoder]
3) mem_sfx: union of (1) and (2)

Also infer dataset and prompt type from the path. Map 'bbox' -> 'box'.
For point prompt, set num_pos_points = 1.

Outputs are written to configs/combo/{index}_{variant}.yaml where index is
1-based order in eval_list.md. If configs/combo/1.yaml exists (example), we
still generate 1_sfx.yaml and 1_mem_sfx.yaml and leave 1.yaml untouched.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_LIST_PATH = REPO_ROOT / "eval_list.md"
OUTPUT_DIR = REPO_ROOT / "configs" / "combo"


def parse_eval_list(md_path: Path) -> List[str]:
    lines = md_path.read_text().splitlines()
    paths: List[str] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # bullet list: "- /path/to/checkpoint.torch"
        if line.startswith("-"):
            # remove leading dash and optional space
            item = line[1:].strip()
            if item:
                paths.append(item)
    return paths


def infer_from_path(path: str) -> Tuple[str, str, str]:
    """
    Infer (dataset, prompt_type, suffix) from a path like:
    .../sam_model_test/cholecseg8k_point_pe/cholecseg8k_point_pe_10.torch

    Returns:
        dataset: e.g., "cholecseg8k"
        prompt_type: one of {"point", "box", "mask"}
        suffix: "pe", "all", or "" if none
    """
    # get the directory name that encodes tokens
    parent = Path(path).parent.name  # e.g., cholecseg8k_point_pe
    tokens = parent.split("_")
    dataset = tokens[0] if tokens else "unknown"
    prompt_raw = tokens[1] if len(tokens) > 1 else "point"
    suffix = tokens[2] if len(tokens) > 2 else ""

    # normalize prompt
    prompt_type = {
        "point": "point",
        "bbox": "box",
        "box": "box",
        "mask": "mask",
    }.get(prompt_raw, "point")

    # only allow pe/all/none for suffix
    if suffix not in ("pe", "all"):
        suffix = ""

    return dataset, prompt_type, suffix


def trainable_modules_for_suffix(suffix: str) -> List[str]:
    # Path-suffix driven modules
    if suffix == "pe":
        return ["mask_decoder", "prompt_encoder"]
    if suffix == "all":
        # all => mask_decoder + prompt_encoder + image_encoder
        return ["mask_decoder", "prompt_encoder", "image_encoder"]
    # default (no suffix)
    return ["mask_decoder"]


def make_yaml_content(
    checkpoint_path: str,
    dataset: str,
    prompt_type: str,
    trainable_modules: List[str],
    combo_name: str,
) -> Dict:
    cfg: Dict = {
        "defaults": [
            f"/data/{dataset}@data",
        ],
        "model": {
            "fintuned_model_path": checkpoint_path,
            "trainable_modules": trainable_modules,
            "prompt_type": prompt_type,
        },
        "combo": {
            "name": combo_name,
        },
        "data_module": {
            "data": "${data}",
        },
    }
    if prompt_type == "point":
        cfg["model"]["num_pos_points"] = 1
    return cfg


def ensure_dataset_combo_data_config(dataset: str) -> None:
    """No-op placeholder retained for compatibility (YAGNI: not creating extra files)."""
    return


def write_yaml(path: Path, content: Dict) -> None:
    header = "# @package _global_\n\n"
    yaml_str = yaml.dump(content, default_flow_style=False, sort_keys=False)
    path.write_text(header + yaml_str + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = parse_eval_list(EVAL_LIST_PATH)
    if not paths:
        raise SystemExit(f"No paths found in {EVAL_LIST_PATH}")

    for idx, ckpt in enumerate(paths, start=1):
        dataset, prompt_type, suffix = infer_from_path(ckpt)
        ensure_dataset_combo_data_config(dataset)

        # Write under dataset-specific folder
        out_dir = OUTPUT_DIR / dataset
        out_dir.mkdir(parents=True, exist_ok=True)

        # Variant 1: memory only
        mem_modules = ["memory_encoder", "memory_attention"]
        combo_name_mem = f"{dataset}_{idx}_mem"
        mem_yaml = make_yaml_content(ckpt, dataset, prompt_type, mem_modules, combo_name_mem)
        # Respect existing example 1.yaml; otherwise write idx_mem.yaml
        mem_path = out_dir / f"{idx}_mem.yaml"
        write_yaml(mem_path, mem_yaml)

        sfx_modules = trainable_modules_for_suffix(suffix)
        # Variant 3: memory + suffix modules (union, preserving order)
        combined = mem_modules + [m for m in sfx_modules if m not in mem_modules]
        combo_name_mem_sfx = f"{dataset}_{idx}_mem_sfx"
        mem_sfx_yaml = make_yaml_content(ckpt, dataset, prompt_type, combined, combo_name_mem_sfx)
        mem_sfx_path = out_dir / f"{idx}_mem_sfx.yaml"
        write_yaml(mem_sfx_path, mem_sfx_yaml)

    print(f"Generated YAMLs for {len(paths)} checkpoints in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
