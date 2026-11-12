#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from semg_fatigue.fast_features import AVAILABLE_MODELS, FeatureBuildConfig, build_feature_cache, run_feature_loso


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fast LOSO sEMG fatigue experiments using domain features.")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--cache-dir", type=str, default="./semg_feature_cache")
    parser.add_argument("--run-dir", type=str, default="./runs/fast_features")
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--stride-sec", type=float, default=0.5)
    parser.add_argument("--use-mvc", action="store_true", default=True)
    parser.add_argument("--no-use-mvc", dest="use_mvc", action="store_false")
    parser.add_argument("--disable-filters", action="store_true", default=False)
    parser.add_argument("--bandpass-low", type=float, default=20.0)
    parser.add_argument("--bandpass-high", type=float, default=450.0)
    parser.add_argument("--notch-hz", type=float, default=50.0)
    parser.add_argument("--include-progress-features", dest="include_progress_features", action="store_true", default=True, help="Add normalized within-trial progress features.")
    parser.add_argument("--no-progress-features", dest="include_progress_features", action="store_false", help="Disable normalized within-trial progress features.")
    parser.add_argument("--subject-limit", type=int, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--models",
        type=str,
        default="xgb_fast",
        help=(
            "Comma-separated model names, or 'all'. "
            f"Available: {', '.join(AVAILABLE_MODELS)}"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-feature-build", action="store_true", help="Reuse an existing feature cache.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    notch_hz = None if args.notch_hz <= 0 else args.notch_hz
    feature_config = FeatureBuildConfig(
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        use_mvc=args.use_mvc,
        disable_filters=args.disable_filters,
        bandpass_low=args.bandpass_low,
        bandpass_high=args.bandpass_high,
        notch_hz=notch_hz,
        include_progress_features=args.include_progress_features,
        subject_limit=args.subject_limit,
        rebuild_cache=args.rebuild_cache,
    )
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "feature_config.json").write_text(json.dumps(feature_config.__dict__, indent=2))

    if not args.skip_feature_build:
        manifest = build_feature_cache(feature_config)
        total_windows = int(manifest.drop_duplicates(["subject_id", "trial_id"])["n_windows"].sum())
        print(f"Built feature cache with {total_windows} windows.")

    raw_model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    model_names = list(AVAILABLE_MODELS) if any(name.lower() == "all" for name in raw_model_names) else raw_model_names
    results = run_feature_loso(
        cache_dir=args.cache_dir,
        run_dir=args.run_dir,
        model_names=model_names,
        seed=args.seed,
    )
    print("\nModel comparison")
    print(json.dumps(results["comparison"], indent=2))
    print(f"Saved outputs to {run_dir.resolve()}")


if __name__ == "__main__":
    main()
