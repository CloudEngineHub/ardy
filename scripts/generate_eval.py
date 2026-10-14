# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate motions with ARDY over a Kimodo Motion Generation Benchmark test suite.

Build the suite for the Ardy motion representation first (in the Kimodo repository):

    python benchmark/create_benchmark.py path/to/testsuite_ardy \\
        --dataset datasets/bones-seed/soma_uniform --motion_rep ardy

then generate here, and embed + evaluate back in the Kimodo repository (its
``benchmark/embed_folder.py`` and ``benchmark/evaluate_folder.py`` operate on the
``motion.npz`` files this script writes). See the README's benchmark section.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from ardy.constraints import FullBodyConstraintSet, load_constraints_lst
from ardy.model import DEFAULT_MODEL, load_model
from ardy.model.load_model import get_env_var
from ardy.model.registry import resolve_model_name
from ardy.motion_rep.tools import length_to_mask
from ardy.postprocess import post_process_motion
from ardy.skeleton import SOMASkeleton30
from ardy.tools import seed_everything, to_numpy


def load_json(path):
    with open(path) as f:
        return json.load(f)


def parse_prompts_from_meta(meta, fps=None):
    """Parse prompt texts and durations (seconds, or frames when ``fps`` is given) from meta.json."""
    if "text" in meta and "duration" in meta:
        duration = float(meta["duration"])
        return [meta["text"]], [int(duration * fps) if fps is not None else duration]
    if "texts" in meta and "durations" in meta:
        durations = [float(d) for d in meta["durations"]]
        if fps is not None:
            durations = [int(d * fps) for d in durations]
        return list(meta["texts"]), durations
    raise ValueError("meta must contain either 'text' and 'duration', or 'texts' and 'durations'.")


def discover_example_folders(root: Path) -> list[tuple[Path, Path]]:
    """Discover leaf directories that contain meta.json.

    Returns list of (src_dir, rel_path).
    """
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Testsuite folder does not exist: {root}")
    out: list[tuple[Path, Path]] = []
    for meta_path in root.rglob("meta.json"):
        src_dir = meta_path.parent
        rel = src_dir.relative_to(root)
        out.append((src_dir, rel))
    return sorted(out, key=lambda x: str(x[1]))


def copy_source_files(src_dir: Path, out_dir: Path) -> None:
    """Copy meta.json, constraints.json, and gt_motion.npz (if present) from src_dir to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("meta.json", "constraints.json", "gt_motion.npz"):
        src_file = src_dir / name
        if src_file.is_file():
            shutil.copy2(src_file, out_dir / name)


class EvalExampleDataset(Dataset):
    """Dataset of example folders: yields text, num_frame, constraints_path (and paths, meta).

    No torch/skeleton in workers so num_workers > 0 is safe with CUDA.
    """

    def __init__(
        self,
        examples: list[tuple[Path, Path]],
        testsuite_root: Path,
        generated_root: Path,
        fps: float,
    ):
        self.examples = examples
        self.testsuite_root = testsuite_root
        self.generated_root = generated_root
        self.fps = fps

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        src_dir, rel_path = self.examples[idx]
        out_dir = self.generated_root / rel_path
        meta_path = src_dir / "meta.json"
        meta = load_json(str(meta_path))
        assert meta.get("num_samples", 1) == 1, "Expected num_samples to be absent or 1 in meta.json"
        texts, durations_sec = parse_prompts_from_meta(meta)
        assert len(texts) == 1, "Expected exactly one prompt (len(texts)==1) per example"
        num_frames = [int(float(d) * self.fps) for d in durations_sec]
        assert len(num_frames) == 1, "Expected exactly one duration per example"
        constraints_path = src_dir / "constraints.json"
        cpath = str(constraints_path) if constraints_path.is_file() else None
        return {
            "rel_path": rel_path,
            "src_dir": str(src_dir),
            "out_dir": str(out_dir),
            "meta": meta,
            "text": texts[0],
            "num_frame": num_frames[0],
            "constraints_path": cpath,
        }


def collate_examples(batch: list[dict]) -> dict[str, Any]:
    """Collate list of example dicts; keep list fields as lists (no stacking)."""
    if not batch:
        return {}
    keys = batch[0].keys()
    out: dict[str, Any] = {}
    for k in keys:
        vals = [b[k] for b in batch]
        out[k] = vals
    return out


def group_by_parent(
    examples: list[tuple[Path, Path]],
) -> list[list[tuple[Path, Path]]]:
    """Group (src_dir, rel_path) by parent directory of rel_path for folder-by-folder processing."""
    from itertools import groupby

    def parent_key(item: tuple[Path, Path]) -> Path:
        rel = item[1]
        return rel.parent if len(rel.parts) > 1 else Path(".")

    sorted_examples = sorted(examples, key=parent_key)
    groups: list[list[tuple[Path, Path]]] = []
    for _key, group in groupby(sorted_examples, key=parent_key):
        groups.append(list(group))
    return groups


def _slice_output_at(output: dict[str, Any], index: int) -> dict[str, Any]:
    """Slice a (possibly nested) output dict at batch index for one sample."""
    out: dict[str, Any] = {}
    for k, v in output.items():
        if isinstance(v, dict):
            out[k] = _slice_output_at(v, index)
        elif isinstance(v, np.ndarray) and v.ndim > 0:
            out[k] = v[index]
        else:
            out[k] = v
    return out


def _crop_output(output: dict[str, Any], num_frames: int) -> dict[str, Any]:
    """Crop a single-sample output dict along the time dimension (axis 0)."""
    out: dict[str, Any] = {}
    for k, v in output.items():
        if isinstance(v, dict):
            out[k] = _crop_output(v, num_frames)
        elif isinstance(v, np.ndarray) and v.ndim >= 1:
            out[k] = v[:num_frames]
        else:
            out[k] = v
    return out


def _default_history_frames(fps: float, gen_horizon_len: int, num_frames_per_token: int) -> int:
    """Longest history that, together with the generation horizon, fits the model's trained 10 s
    window (same budget as ``ardy/scripts/inference/generate.py``)."""
    max_window_len = (int(10 * fps) // num_frames_per_token) * num_frames_per_token
    return ((max_window_len - gen_horizon_len) // num_frames_per_token) * num_frames_per_token


def parse_args():
    parser = argparse.ArgumentParser(description="Recursively generate motions with Ardy from a testsuite folder tree")
    parser.add_argument("--benchmark", type=str, default="testsuite", help="Root folder containing meta.json leaves")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output root; the hierarchy is mirrored here. If omitted, motions are generated in place.",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers (default: 4)")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Ardy model: short key (core/g1/soma), released folder name, or a full path to a checkpoint folder.",
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default=None,
        help="Local dir holding released model folders (falls back to CHECKPOINTS_DIR; else HF download).",
    )
    parser.add_argument(
        "--diffusion_steps",
        type=int,
        default=None,
        help="Denoising steps, at most the model's num_base_steps (the default). Not read from meta.json.",
    )
    parser.add_argument(
        "--cfg_weight",
        type=float,
        nargs="+",
        default=[2.0, 2.0],
        help="CFG weight(s): one float (text) or two floats (text, constraint). Default: 2.0 2.0",
    )
    parser.add_argument(
        "--history_frames",
        type=int,
        default=None,
        help="History frames visible per autoregressive step (multiple of the token size). "
        "Default: longest history fitting the trained 10 s window.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed used when meta.json has none (default: 0)")
    parser.add_argument(
        "--fullbody_rotations",
        action="store_true",
        help="Full-body keyframes also constrain every joint's global rotation. Required for models trained "
        "with motiongen's full-row FullbodyKeyframe masks (ardy runs a627c36..2026-09-04); leave off for the "
        "released ARDY models (positions-only contract).",
    )
    parser.add_argument("--postprocess", action="store_true", help="Apply foot-skate post-processing (off for eval)")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate even if motion.npz exists")
    parser.add_argument("--text_encoder_fp32", action="store_true", help="fp32 text encoder instead of bf16")
    return parser.parse_args()


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    args = parse_args()
    testsuite_root = Path(args.benchmark).resolve()
    generated_root = Path(args.output).resolve() if args.output is not None else testsuite_root
    in_place = generated_root == testsuite_root

    if len(args.cfg_weight) == 1:
        cfg_weight = float(args.cfg_weight[0])
    elif len(args.cfg_weight) == 2:
        cfg_weight = (float(args.cfg_weight[0]), float(args.cfg_weight[1]))
    else:
        raise ValueError("--cfg_weight expects one float (text) or two floats (text, constraint).")

    examples = discover_example_folders(testsuite_root)
    if not examples:
        raise SystemExit(f"No folders with meta.json found under {testsuite_root}")
    print(f"Discovered {len(examples)} example folders.")

    checkpoints_dir = args.checkpoints_dir or get_env_var("CHECKPOINTS_DIR")
    if Path(args.model).is_dir():
        # a path to a checkpoint folder: its parent acts as the local checkpoints dir
        model_dir = Path(args.model).resolve()
        checkpoints_dir, resolved_model = str(model_dir.parent), model_dir.name
    else:
        resolved_model = resolve_model_name(args.model, checkpoints_dir=checkpoints_dir)
    model = load_model(
        resolved_model,
        device=device,
        checkpoints_dir=checkpoints_dir,
        text_encoder_fp32=args.text_encoder_fp32,
    )
    print(f"Generating with model: {resolved_model}")
    fps = model.motion_rep.fps

    num_base_steps = int(model.diffusion.num_base_steps)
    diffusion_steps = args.diffusion_steps if args.diffusion_steps is not None else num_base_steps
    if not 1 <= diffusion_steps <= num_base_steps:
        raise ValueError(f"--diffusion_steps must be between 1 and {num_base_steps}; got {diffusion_steps}.")

    patch = model.num_frames_per_token
    history_frames = args.history_frames
    if history_frames is None:
        history_frames = _default_history_frames(fps, model.gen_horizon_len, patch)
    elif history_frames < patch or history_frames % patch != 0:
        raise ValueError(f"--history_frames must be a positive multiple of {patch}.")
    print(
        f"{diffusion_steps} denoising steps, {history_frames} history frames, cfg_weight={cfg_weight}, "
        f"fullbody rotations={'on' if args.fullbody_rotations else 'off'}"
    )

    is_soma30 = isinstance(model.skeleton, SOMASkeleton30)
    device_t = torch.device(device)

    groups = group_by_parent(examples)
    total_generated = 0
    total_skipped = 0
    total_examples = len(examples)
    for group in groups:
        rel_path_0 = group[0][1]
        folder_label = str(rel_path_0.parent) if rel_path_0.parent != Path(".") else testsuite_root.name
        num_in_folder = len(group)
        print(f"Generating folder: {folder_label} ({num_in_folder} motions)")

        dataset = EvalExampleDataset(group, testsuite_root, generated_root, fps=fps)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_examples,
        )

        folder_generated = 0
        folder_skipped = 0
        for batch in loader:
            if args.overwrite:
                selected = list(range(len(batch["rel_path"])))
            else:
                selected = []
                for i, out_dir_str in enumerate(batch["out_dir"]):
                    if (Path(out_dir_str) / "motion.npz").is_file():
                        folder_skipped += 1
                        total_skipped += 1
                    else:
                        selected.append(i)
            if not selected:
                continue

            src_dirs = [batch["src_dir"][i] for i in selected]
            out_dirs = [batch["out_dir"][i] for i in selected]
            metas = [batch["meta"][i] for i in selected]
            texts = [batch["text"][i] for i in selected]
            num_frames_lst = [batch["num_frame"][i] for i in selected]
            constraints_paths = [batch["constraints_path"][i] for i in selected]

            constraints_lst = [
                load_constraints_lst(cpath, model.skeleton) if cpath else [] for cpath in constraints_paths
            ]
            for c in (c for c_lst in constraints_lst for c in c_lst if isinstance(c, FullBodyConstraintSet)):
                c.constrain_rotations = args.fullbody_rotations
            for c_lst, n in zip(constraints_lst, num_frames_lst):
                for c in c_lst:
                    if int(c.frame_indices.max()) >= n:
                        raise ValueError(f"Constraint frame {int(c.frame_indices.max())} beyond motion length {n}.")

            if not in_place:
                for src_dir, out_dir in zip(src_dirs, out_dirs):
                    copy_source_files(Path(src_dir), Path(out_dir))

            # Same convention as generate_eval.py: the first example's seed seeds the whole batch.
            seed_everything(metas[0].get("seed", args.seed))

            batch_size = len(texts)
            lengths = torch.tensor(num_frames_lst, device=device_t)
            num_frames = int(lengths.max())
            pad_mask = length_to_mask(lengths)
            first_heading_angle = torch.zeros(batch_size, device=device_t)  # facing +Z

            observed_motion, motion_mask = None, None
            if any(constraints_lst):
                observed_motion, motion_mask = model.motion_rep.create_conditions_from_constraints_batched(
                    constraints_lst, lengths, to_normalize=True, device=device
                )
                # The denoiser blends `x * (1 - mask) + observed * mask`, i.e. it needs a FLOAT mask and
                # only reads observed cells where mask == 1 -- the training-side create_mask_cond
                # contract. ArdyMotionRep already returns that; KimodoMotionRep (smooth-root ardy
                # models) returns a bool mask with unobserved cells left as normalized zeros.
                motion_mask = motion_mask.to(observed_motion.dtype)
                observed_motion = observed_motion * motion_mask

            bar_desc = (
                f"  Generated {folder_generated} / {num_in_folder} "
                f"(skipped: {folder_skipped}) (total: {total_generated + total_skipped} / {total_examples})"
            )
            with torch.no_grad():
                motion = model(
                    texts,
                    num_frames,
                    num_denoising_steps=diffusion_steps,
                    pad_mask=pad_mask,
                    first_heading_angle=first_heading_angle,
                    motion_mask=motion_mask,
                    observed_motion=observed_motion,
                    cfg_weight=cfg_weight,
                    crop_history_length=history_frames,
                    progress_bar=lambda x: tqdm(x, leave=False, desc=bar_desc),
                )
                output = model.motion_rep.inverse(motion, is_normalized=True)

                if args.postprocess:
                    # post-processing is per sample (constraints differ across the batch)
                    corrected = [
                        post_process_motion(
                            output["local_rot_mats"][b : b + 1, :n],
                            output["root_positions"][b : b + 1, :n],
                            output["foot_contacts"][b : b + 1, :n],
                            model.skeleton,
                            constraint_lst=constraints_lst[b] or None,
                        )
                        for b, n in enumerate(num_frames_lst)
                    ]
                    for key in corrected[0]:
                        padded = output[key].clone()
                        for b, n in enumerate(num_frames_lst):
                            padded[b, :n] = corrected[b][key][0]
                        output[key] = padded

                if is_soma30:
                    output = model.skeleton.output_to_SOMASkeleton77(output)
                output = to_numpy(output)

            for b, out_dir in enumerate(out_dirs):
                sample_output = _crop_output(_slice_output_at(output, b), num_frames_lst[b])
                np.savez(Path(out_dir) / "motion.npz", **sample_output)
                total_generated += 1
                folder_generated += 1

            print(
                f"\r  Generated {folder_generated} / {num_in_folder} (skipped: {folder_skipped}) "
                f"(total: {total_generated + total_skipped} / {total_examples})",
                end="",
                flush=True,
            )

        print()
        print(
            f"  Finished folder {folder_label} ({num_in_folder} motions, generated: {folder_generated}, skipped: {folder_skipped})."
        )

    where = testsuite_root if in_place else generated_root
    print(f"Generated {total_generated} motions under {where}.")


if __name__ == "__main__":
    main()
