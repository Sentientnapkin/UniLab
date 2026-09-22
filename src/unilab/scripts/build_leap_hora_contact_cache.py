"""Build a contact-aware HORA Stage A grasp-cache pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from unilab.scripts.qualify_leap_hora_grasps import (
    compose_qualification_config,
    qualification_plan,
    run_qualification,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-control-steps", type=int, required=True)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("src/unilab/assets/caches/hora/leap_precision_grasps_v2.npz"),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("src/unilab/assets/caches/hora/leap_precision_grasps_v4_contact_pilot"),
    )
    parser.add_argument("--cache-size", type=int, default=1000)
    parser.add_argument("--anchor-fraction", type=float, default=0.8)
    parser.add_argument("--middle-contact-fraction", type=float, default=0.4)
    parser.add_argument("--minimum-raw-thumb-contact", type=float, default=0.5)
    parser.add_argument("--minimum-raw-middle-contact", type=float, default=0.5)
    parser.add_argument("--minimum-raw-opposing-contact", type=float, default=0.5)
    parser.add_argument("--minimum-raw-opposing-fingers", type=int, default=2)
    parser.add_argument("--anchor-minimum-seconds", type=float, default=16.0)
    parser.add_argument("--recoverable-minimum-seconds", type=float, default=1.0)
    parser.add_argument("--marginal-minimum-seconds", type=float, default=0.5)
    parser.add_argument("--describe-only", action="store_true")
    return parser


def compose_contact_cache_config(args: argparse.Namespace):
    """Compose passive qualification with contact-aware selection parameters."""
    for name in (
        "middle_contact_fraction",
        "minimum_raw_thumb_contact",
        "minimum_raw_middle_contact",
        "minimum_raw_opposing_contact",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    if not 1 <= args.minimum_raw_opposing_fingers <= 3:
        raise ValueError("--minimum-raw-opposing-fingers must be in [1, 3]")

    cfg = compose_qualification_config(
        num_envs=args.num_envs,
        seed=args.seed,
        input_path=args.input,
        output_prefix=args.output_prefix,
        anchor_minimum_seconds=args.anchor_minimum_seconds,
        recoverable_minimum_seconds=args.recoverable_minimum_seconds,
        marginal_minimum_seconds=args.marginal_minimum_seconds,
        stage_a_size=args.cache_size,
        stage_a_anchor_fraction=args.anchor_fraction,
    )
    params = "env.recorders.passive_qualification.params"
    OmegaConf.update(
        cfg,
        f"{params}.training_middle_contact_fraction",
        args.middle_contact_fraction,
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        f"{params}.minimum_raw_thumb_contact_fraction",
        args.minimum_raw_thumb_contact,
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        f"{params}.minimum_raw_middle_contact_fraction",
        args.minimum_raw_middle_contact,
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        f"{params}.minimum_raw_opposing_contact_fraction",
        args.minimum_raw_opposing_contact,
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        f"{params}.minimum_raw_opposing_fingers",
        args.minimum_raw_opposing_fingers,
        force_add=True,
    )
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cfg = compose_contact_cache_config(args)
    plan = qualification_plan(cfg, max_control_steps=args.max_control_steps)
    contact_selection = {
        "cache_size": args.cache_size,
        "anchor_fraction": args.anchor_fraction,
        "middle_contact_fraction": args.middle_contact_fraction,
        "minimum_raw_thumb_contact_fraction": args.minimum_raw_thumb_contact,
        "minimum_raw_middle_contact_fraction": args.minimum_raw_middle_contact,
        "minimum_raw_opposing_contact_fraction": args.minimum_raw_opposing_contact,
        "minimum_raw_opposing_fingers": args.minimum_raw_opposing_fingers,
    }
    print(
        json.dumps(
            {
                "qualification_plan": plan.as_dict(),
                "contact_selection": contact_selection,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.describe_only:
        return 0
    result = run_qualification(cfg, max_control_steps=args.max_control_steps)
    print(json.dumps({"qualification_result": result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
