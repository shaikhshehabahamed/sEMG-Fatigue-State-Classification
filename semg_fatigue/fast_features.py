from __future__ import annotations

import inspect
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import kurtosis, skew
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, matthews_corrcoef
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .data import (
    apply_mvc_normalization,
    canonicalize_channel_order,
    compute_mvc_map,
    load_emg_trial,
    load_label_file,
    preprocess_signal,
    resolve_label_csv,
    subject_sort_key,
)

AVAILABLE_MODELS = (
    "lda",
    "logistic_regression",
    "svm_rbf",
    "random_forest",
    "xgb_fast",
)

MODEL_DISPLAY_NAMES = {
    "lda": "LDA",
    "logistic_regression": "Logistic Regression",
    "svm_rbf": "SVM-RBF",
    "random_forest": "Random Forest",
    "xgb_fast": "XGBoost",
}


@dataclass
class FeatureBuildConfig:
    data_root: str = "./data"
    cache_dir: str = "./semg_feature_cache"
    window_sec: float = 1.0
    stride_sec: float = 0.5
    use_mvc: bool = True
    disable_filters: bool = False
    bandpass_low: float = 20.0
    bandpass_high: float = 450.0
    notch_hz: Optional[float] = 50.0
    include_progress_features: bool = True
    subject_limit: Optional[int] = None
    rebuild_cache: bool = False


def split_subjects_for_fold(
    subjects: Sequence[str],
    test_subject: str,
    fold_index: int,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    train_subjects = [s for s in subjects if s != test_subject]
    if len(train_subjects) <= 1:
        return train_subjects, [], [test_subject]

    # Seeded shuffle: reproducible but different for every fold so every
    # subject gets roughly equal validation exposure across folds.
    rng = np.random.default_rng(seed + fold_index)
    shuffled = list(train_subjects)
    rng.shuffle(shuffled)

    n_val = min(2, len(shuffled) - 1)
    val_subjects = shuffled[:n_val]
    final_train = shuffled[n_val:]
    return final_train, val_subjects, [test_subject]


def _nearest_labels(label_times: np.ndarray, labels: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    if len(label_times) == 0 or len(labels) == 0:
        raise ValueError("Cannot assign labels from an empty label series.")
    if len(label_times) != len(labels):
        raise ValueError("label_times and labels must have the same length.")
    if len(label_times) == 1:
        return np.full(len(query_times), labels[0], dtype=np.int64)
    indices = np.searchsorted(label_times, query_times)
    indices = np.clip(indices, 1, len(label_times) - 1)
    left = indices - 1
    right = indices
    choose_left = np.abs(label_times[left] - query_times) <= np.abs(label_times[right] - query_times)
    nearest = np.where(choose_left, left, right)
    return labels[nearest].astype(np.int64, copy=False)


def _majority_vote_labels(
    label_times: np.ndarray,
    labels: np.ndarray,
    start_times: np.ndarray,
    end_times: np.ndarray,
    n_samples: int = 7,
) -> np.ndarray:
    """Assign each window the majority label among n_samples evenly-spaced query points."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    fracs = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)
    query = start_times[:, None] + (end_times - start_times)[:, None] * fracs[None, :]
    flat_labels = _nearest_labels(label_times, labels, query.ravel())
    vote_matrix = flat_labels.reshape(len(start_times), n_samples)
    num_classes = int(labels.max()) + 1 if len(labels) > 0 else 1
    one_hot = (vote_matrix[:, :, None] == np.arange(num_classes, dtype=np.int64)[None, None, :])
    return one_hot.sum(axis=1).argmax(axis=1).astype(np.int64)


def _bandpower(power: np.ndarray, freqs: np.ndarray, low: float, high: float) -> np.ndarray:
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return np.zeros(power.shape[:2], dtype=np.float32)
    return power[:, :, mask].sum(axis=2).astype(np.float32)


def _feature_names(channel_names: Sequence[str], include_progress_features: bool = False) -> List[str]:
    per_channel = [
        "mean",
        "std",
        "rms",
        "mav",
        "iemg",
        "max_abs",
        "p2p",
        "q25_abs",
        "q50_abs",
        "q75_abs",
        "waveform_len",
        "slope_rms",
        "zero_cross_rate",
        "slope_sign_change",
        "hjorth_mobility",
        "hjorth_complexity",
        "skew",
        "kurtosis",
        "total_power",
        "mean_freq",
        "median_freq",
        "peak_freq",
        "spectral_entropy",
        "bp_20_60",
        "bp_60_150",
        "bp_150_250",
        "bp_250_450",
        "bp_high_low_ratio",
    ]
    names: List[str] = []
    cleaned = [name.replace(" ", "_").replace(":", "").replace("/", "_") for name in channel_names]
    for channel in cleaned:
        names.extend([f"{channel}__{feature}" for feature in per_channel])
    for i in range(len(cleaned)):
        for j in range(i + 1, len(cleaned)):
            names.append(f"{cleaned[i]}__{cleaned[j]}__corr")
    names.extend([
        "global_rms_mean",
        "global_rms_std",
        "global_mav_mean",
        "global_waveform_len_mean",
        "global_median_freq_mean",
        "global_median_freq_std",
        "global_high_low_ratio_mean",
    ])
    if include_progress_features:
        names.extend(["trial_progress", "trial_progress_sq", "trial_progress_cube"])
    return names


def extract_trial_features(
    signal: np.ndarray,
    time_axis: np.ndarray,
    fs: float,
    label_times: np.ndarray,
    labels: np.ndarray,
    channel_names: Sequence[str],
    window_sec: float,
    stride_sec: float,
    include_progress_features: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    if window_sec <= 0 or stride_sec <= 0:
        raise ValueError("window_sec and stride_sec must be positive.")
    if len(signal) != len(time_axis):
        raise ValueError("signal and time_axis must contain the same number of samples.")
    if len(label_times) == 0 or len(labels) == 0:
        raise ValueError("label_times and labels must not be empty.")
    if len(label_times) != len(labels):
        raise ValueError("label_times and labels must have the same length.")
    samples_per_window = max(1, int(round(window_sec * fs)))
    stride_samples = max(1, int(round(stride_sec * fs)))
    if len(signal) < samples_per_window:
        return (
            np.empty((0, len(_feature_names(channel_names, include_progress_features))), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
            _feature_names(channel_names, include_progress_features),
        )

    starts = np.arange(0, len(signal) - samples_per_window + 1, stride_samples, dtype=np.int64)
    ends = starts + samples_per_window - 1
    start_times = time_axis[starts].astype(np.float32)
    end_times = time_axis[ends].astype(np.float32)
    center_times = (start_times + end_times) * np.float32(0.5)
    y = _majority_vote_labels(label_times, labels, start_times, end_times)

    # Shape: [windows, channels, samples]
    windows = np.lib.stride_tricks.sliding_window_view(signal, samples_per_window, axis=0)[starts]
    windows = windows.astype(np.float32, copy=False)
    eps = np.float32(1e-8)

    abs_windows = np.abs(windows)
    mean = windows.mean(axis=2)
    std = windows.std(axis=2)
    rms = np.sqrt(np.mean(windows * windows, axis=2) + eps)
    mav = abs_windows.mean(axis=2)
    iemg = abs_windows.sum(axis=2) / np.float32(fs)
    max_abs = abs_windows.max(axis=2)
    p2p = windows.max(axis=2) - windows.min(axis=2)
    quantiles = np.quantile(abs_windows, [0.25, 0.50, 0.75], axis=2).astype(np.float32)

    diff1 = np.diff(windows, axis=2)
    diff2 = np.diff(diff1, axis=2)
    waveform_len = np.mean(np.abs(diff1), axis=2)
    slope_rms = np.sqrt(np.mean(diff1 * diff1, axis=2) + eps)
    zero_cross = ((windows[:, :, 1:] * windows[:, :, :-1]) < 0).mean(axis=2).astype(np.float32)
    slope_sign = ((diff1[:, :, 1:] * diff1[:, :, :-1]) < 0).mean(axis=2).astype(np.float32)
    mobility = np.sqrt((diff1.var(axis=2) + eps) / (windows.var(axis=2) + eps))
    diff_mobility = np.sqrt((diff2.var(axis=2) + eps) / (diff1.var(axis=2) + eps))
    complexity = diff_mobility / (mobility + eps)
    skewness = skew(windows, axis=2, bias=False, nan_policy="omit").astype(np.float32)
    kurt = kurtosis(windows, axis=2, fisher=True, bias=False, nan_policy="omit").astype(np.float32)
    skewness = np.nan_to_num(skewness, copy=False)
    kurt = np.nan_to_num(kurt, copy=False)

    taper = np.hanning(samples_per_window).astype(np.float32)
    spec = np.fft.rfft(windows * taper.reshape(1, 1, -1), axis=2)
    power = (np.abs(spec) ** 2).astype(np.float32)
    freqs = np.fft.rfftfreq(samples_per_window, d=1.0 / fs).astype(np.float32)
    valid = (freqs >= 20.0) & (freqs <= 450.0)
    if not np.any(valid):
        raise ValueError(
            "No FFT frequency bins fall in the 20-450 Hz EMG band; "
            "increase fs/window_sec or disable frequency-domain features."
        )
    power = power[:, :, valid]
    freqs = freqs[valid]

    total_power = power.sum(axis=2) + eps
    mean_freq = (power * freqs.reshape(1, 1, -1)).sum(axis=2) / total_power
    cumsum_power = np.cumsum(power, axis=2)
    median_target = total_power[:, :, None] * 0.5
    median_indices = (cumsum_power < median_target).sum(axis=2)
    median_indices = np.clip(median_indices, 0, len(freqs) - 1)
    median_freq = freqs[median_indices]
    peak_freq = freqs[np.argmax(power, axis=2)]
    prob_power = power / total_power[:, :, None]
    entropy_norm = np.log(power.shape[2]) if power.shape[2] > 1 else 1.0
    spectral_entropy = -(prob_power * np.log(prob_power + eps)).sum(axis=2) / entropy_norm
    bp_20_60 = _bandpower(power, freqs, 20.0, 60.0)
    bp_60_150 = _bandpower(power, freqs, 60.0, 150.0)
    bp_150_250 = _bandpower(power, freqs, 150.0, 250.0)
    bp_250_450 = _bandpower(power, freqs, 250.0, 450.0)
    high_low_ratio = (bp_150_250 + bp_250_450) / (bp_20_60 + bp_60_150 + eps)

    per_channel_blocks = [
        mean,
        std,
        rms,
        mav,
        iemg,
        max_abs,
        p2p,
        quantiles[0],
        quantiles[1],
        quantiles[2],
        waveform_len,
        slope_rms,
        zero_cross,
        slope_sign,
        mobility,
        complexity,
        skewness,
        kurt,
        total_power,
        mean_freq,
        median_freq,
        peak_freq,
        spectral_entropy,
        bp_20_60,
        bp_60_150,
        bp_150_250,
        bp_250_450,
        high_low_ratio,
    ]
    per_channel = np.stack(per_channel_blocks, axis=2).reshape(len(starts), -1)

    corr_features = []
    for i in range(windows.shape[1]):
        xi = windows[:, i, :]
        xi = xi - xi.mean(axis=1, keepdims=True)
        xi_std = np.sqrt(np.mean(xi * xi, axis=1) + eps)
        for j in range(i + 1, windows.shape[1]):
            xj = windows[:, j, :]
            xj = xj - xj.mean(axis=1, keepdims=True)
            xj_std = np.sqrt(np.mean(xj * xj, axis=1) + eps)
            corr_features.append(np.mean(xi * xj, axis=1) / (xi_std * xj_std + eps))
    corr = np.stack(corr_features, axis=1).astype(np.float32) if corr_features else np.empty((len(starts), 0), dtype=np.float32)

    global_features = np.stack(
        [
            rms.mean(axis=1),
            rms.std(axis=1),
            mav.mean(axis=1),
            waveform_len.mean(axis=1),
            median_freq.mean(axis=1),
            median_freq.std(axis=1),
            high_low_ratio.mean(axis=1),
        ],
        axis=1,
    ).astype(np.float32)

    feature_blocks = [per_channel.astype(np.float32), corr, global_features]
    if include_progress_features:
        trial_start = float(time_axis[0])
        duration = float(time_axis[-1]) - trial_start
        if duration <= 0:
            duration = float(len(time_axis)) / fs
        progress = np.clip(
            (center_times.astype(np.float32) - np.float32(trial_start)) / np.float32(max(duration, 1e-6)),
            0.0, 1.0,
        )
        progress_features = np.stack([progress, progress ** 2, progress ** 3], axis=1).astype(np.float32)
        feature_blocks.append(progress_features)
    x = np.concatenate(feature_blocks, axis=1)
    x = np.nan_to_num(x, copy=False, posinf=0.0, neginf=0.0).astype(np.float32)
    return x, y, center_times, _feature_names(channel_names, include_progress_features)


def build_feature_cache(config: FeatureBuildConfig) -> pd.DataFrame:
    data_root = Path(config.data_root)
    emg_root = data_root / "sEMG_data"
    label_root = data_root / "self_perceived_fatigue_index"
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    subjects = sorted([p.name for p in emg_root.glob("subject_*") if p.is_dir()], key=subject_sort_key)
    if config.subject_limit is not None:
        subjects = subjects[: int(config.subject_limit)]
    if not subjects:
        raise FileNotFoundError(f"No subjects found under {emg_root}")

    manifest_rows: List[Dict[str, object]] = []
    common_feature_names: Optional[List[str]] = None
    config_payload = asdict(config)

    for subject_id in subjects:
        subject_start = time.time()
        subject_emg_dir = emg_root / subject_id
        subject_label_dir = label_root / subject_id
        mvc_map = compute_mvc_map(subject_emg_dir / "MVC") if config.use_mvc else {}
        subject_windows = 0

        for trial_csv in sorted(subject_emg_dir.glob("trial_*.csv")):
            trial_id = trial_csv.stem
            label_csv = resolve_label_csv(subject_label_dir, trial_id)
            if label_csv is None:
                continue

            out_path = cache_dir / f"{subject_id}_{trial_id}_features.npz"
            meta_path = cache_dir / f"{subject_id}_{trial_id}_features.json"
            if out_path.exists() and meta_path.exists() and not config.rebuild_cache:
                meta = json.loads(meta_path.read_text())
                if meta.get("config") == config_payload:
                    n_windows = int(meta["n_windows"])
                    labels = np.asarray(meta["label_counts"], dtype=np.int64)
                    for label_value, count in enumerate(labels.tolist()):
                        if count:
                            manifest_rows.append({
                                "subject_id": subject_id,
                                "trial_id": trial_id,
                                "feature_path": str(out_path),
                                "n_windows": n_windows,
                                "label": label_value,
                                "label_count": int(count),
                            })
                    common_feature_names = list(meta["feature_names"])
                    subject_windows += n_windows
                    continue

            time_axis, signal, channel_names, fs = load_emg_trial(trial_csv)
            signal = preprocess_signal(
                signal=signal,
                fs=fs,
                apply_filters=not config.disable_filters,
                bandpass_low=float(config.bandpass_low),
                bandpass_high=float(config.bandpass_high),
                notch_hz=config.notch_hz,
            )
            if config.use_mvc and mvc_map:
                signal = apply_mvc_normalization(signal, channel_names, mvc_map)
            signal, channel_names = canonicalize_channel_order(signal, channel_names)
            label_times, labels = load_label_file(label_csv)

            x, y, center_times, feature_names = extract_trial_features(
                signal=signal,
                time_axis=time_axis,
                fs=fs,
                label_times=label_times,
                labels=labels,
                channel_names=channel_names,
                window_sec=float(config.window_sec),
                stride_sec=float(config.stride_sec),
                include_progress_features=bool(config.include_progress_features),
            )
            if common_feature_names is None:
                common_feature_names = feature_names
            elif common_feature_names != feature_names:
                raise ValueError(f"Feature names changed at {trial_csv}")

            np.savez_compressed(out_path, X=x, y=y, center_times=center_times.astype(np.float32))
            label_counts = np.bincount(y, minlength=int(labels.max()) + 1).astype(int).tolist() if len(y) else []
            meta_path.write_text(json.dumps({
                "subject_id": subject_id,
                "trial_id": trial_id,
                "source_csv": str(trial_csv),
                "label_csv": str(label_csv),
                "n_windows": int(len(y)),
                "label_counts": label_counts,
                "feature_names": feature_names,
                "fs": float(fs),
                "config": config_payload,
            }, indent=2))
            for label_value, count in enumerate(label_counts):
                if count:
                    manifest_rows.append({
                        "subject_id": subject_id,
                        "trial_id": trial_id,
                        "feature_path": str(out_path),
                        "n_windows": int(len(y)),
                        "label": int(label_value),
                        "label_count": int(count),
                    })
            subject_windows += len(y)

        print(f"[features] {subject_id}: {subject_windows} windows in {time.time() - subject_start:.1f}s")

    manifest = pd.DataFrame(manifest_rows)
    if manifest.empty:
        raise RuntimeError("Feature manifest is empty.")
    manifest.to_csv(cache_dir / "feature_manifest.csv", index=False)
    if common_feature_names is not None:
        (cache_dir / "feature_names.json").write_text(json.dumps(common_feature_names, indent=2))
    return manifest


def load_feature_table(cache_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    cache_path = Path(cache_dir)
    feature_files = sorted(cache_path.glob("*_features.npz"))
    if not feature_files:
        raise FileNotFoundError(f"No feature files found in {cache_path}")

    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    subject_parts: List[np.ndarray] = []
    trial_parts: List[np.ndarray] = []
    for path in feature_files:
        stem = path.name.replace("_features.npz", "")
        subject_id, trial_id = stem.rsplit("_trial_", 1)
        trial_id = f"trial_{trial_id}"
        with np.load(path) as blob:
            x = blob["X"].astype(np.float32)
            y = blob["y"].astype(np.int64)
        x_parts.append(x)
        y_parts.append(y)
        subject_parts.append(np.full(len(y), subject_id, dtype=object))
        trial_parts.append(np.full(len(y), trial_id, dtype=object))
    feature_names_path = cache_path / "feature_names.json"
    if feature_names_path.exists():
        feature_names = json.loads(feature_names_path.read_text())
    else:
        import warnings as _w
        _w.warn(f"feature_names.json not found in {cache_path}; feature names will be generic (f0, f1, …). Rebuild the cache to fix this.")
        feature_names = [f"f{i}" for i in range(x_parts[0].shape[1])]
    return (
        np.concatenate(x_parts, axis=0),
        np.concatenate(y_parts, axis=0),
        np.concatenate(subject_parts, axis=0),
        np.concatenate(trial_parts, axis=0),
        feature_names,
    )


def _available_model_text() -> str:
    return ", ".join(AVAILABLE_MODELS)


def make_model(model_name: str, seed: int, num_classes: int):
    if model_name == "lda":
        return make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        )
    if model_name == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                solver="lbfgs",
                max_iter=5000,
                random_state=seed,
            ),
        )
    if model_name == "svm_rbf":
        return make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(
                estimator=SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    cache_size=1000,
                    random_state=seed,
                ),
                method="sigmoid",
                cv=3,
                ensemble=False,
                n_jobs=-1,
            ),
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        )
    if model_name == "xgb_fast":
        return XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            tree_method="hist",
            n_estimators=260,
            max_depth=3,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=4.0,
            reg_lambda=2.0,
            reg_alpha=0.05,
            importance_type="gain",
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(f"Unknown model_name: {model_name}. Available models: {_available_model_text()}")


def _final_estimator(model):
    if isinstance(model, Pipeline):
        return model.steps[-1][1]
    return model


def _fit_accepts_sample_weight(estimator) -> bool:
    try:
        return "sample_weight" in inspect.signature(estimator.fit).parameters
    except (TypeError, ValueError):
        return False


def _fit_model(model, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray):
    if isinstance(model, Pipeline):
        final_step_name, final_estimator = model.steps[-1]
        if _fit_accepts_sample_weight(final_estimator):
            return model.fit(x, y, **{f"{final_step_name}__sample_weight": sample_weight})
        return model.fit(x, y)
    if _fit_accepts_sample_weight(model):
        return model.fit(x, y, sample_weight=sample_weight)
    return model.fit(x, y)


def _model_classes(model) -> Optional[np.ndarray]:
    estimator = _final_estimator(model)
    classes = getattr(estimator, "classes_", None)
    if classes is None:
        classes = getattr(model, "classes_", None)
    if classes is None:
        return None
    return np.asarray(classes, dtype=np.int64)


def _predict_proba(model, x: np.ndarray, num_classes: int) -> np.ndarray:
    proba = np.asarray(model.predict_proba(x), dtype=np.float64)
    if proba.ndim == 1:
        if num_classes != 2:
            raise ValueError(f"Expected {num_classes} probability columns, got a 1D prediction.")
        proba = np.stack([1.0 - proba, proba], axis=1)
    classes = _model_classes(model)
    if classes is not None and len(classes) == proba.shape[1]:
        fixed = np.zeros((len(proba), num_classes), dtype=np.float64)
        for source_idx, class_id in enumerate(classes):
            if 0 <= int(class_id) < num_classes:
                fixed[:, int(class_id)] = proba[:, source_idx]
        proba = fixed
    elif proba.shape[1] != num_classes:
        fixed = np.zeros((len(proba), num_classes), dtype=np.float64)
        width = min(num_classes, proba.shape[1])
        fixed[:, :width] = proba[:, :width]
        proba = fixed
    row_sum = proba.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0] = 1.0
    return proba / row_sum


def monotonic_fatigue_decode(proba: np.ndarray) -> np.ndarray:
    """Decode one trial as a non-decreasing 0 -> 1 -> 2 sequence."""
    if len(proba) == 0:
        return np.empty((0,), dtype=np.int64)
    if proba.shape[1] != 3:
        import warnings as _w
        _w.warn(f"monotonic_fatigue_decode expects 3 classes, got {proba.shape[1]}; falling back to argmax.")
        return np.argmax(proba, axis=1).astype(np.int64)
    logp = np.log(np.clip(proba, 1e-8, 1.0))
    n = len(logp)
    prefix0 = np.concatenate([[0.0], np.cumsum(logp[:, 0])])
    prefix1 = np.concatenate([[0.0], np.cumsum(logp[:, 1])])
    prefix2 = np.concatenate([[0.0], np.cumsum(logp[:, 2])])
    # score(t1) = prefix0[t1] + max_{t2>=t1}(prefix1[t2] - prefix2[t2]) + prefix2[n] - prefix1[t1]
    # Computed in O(n) via suffix-maximum of (prefix1 - prefix2).
    combined = prefix1 - prefix2
    suffix_max = np.maximum.accumulate(combined[::-1])[::-1]
    scores = prefix0 - prefix1 + float(prefix2[n]) + suffix_max
    best_t1 = int(np.argmax(scores))
    best_t2 = best_t1 + int(np.argmax(combined[best_t1:]))
    pred = np.empty(n, dtype=np.int64)
    pred[:best_t1] = 0
    pred[best_t1:best_t2] = 1
    pred[best_t2:] = 2
    return pred


def decode_by_trial(proba: np.ndarray, trial_ids: np.ndarray) -> np.ndarray:
    decoded = np.empty((len(proba),), dtype=np.int64)
    for trial_id in sorted(set(trial_ids.tolist())):
        mask = trial_ids == trial_id
        decoded[mask] = monotonic_fatigue_decode(proba[mask])
    return decoded


def make_group_ids(subject_ids: np.ndarray, trial_ids: np.ndarray) -> np.ndarray:
    return np.asarray([f"{s}|{t}" for s, t in zip(subject_ids.tolist(), trial_ids.tolist())], dtype=object)


def fit_progress_thresholds(progress: np.ndarray, y: np.ndarray, group_ids: np.ndarray) -> Tuple[float, float]:
    p01: List[float] = []
    p12: List[float] = []
    for group_id in sorted(set(group_ids.tolist())):
        mask = group_ids == group_id
        yy = y[mask]
        pp = progress[mask]
        changes = np.flatnonzero(yy[1:] != yy[:-1]) + 1
        if len(changes) >= 2:
            p01.append(float(pp[changes[0]]))
            p12.append(float(pp[changes[1]]))
    if not p01 or not p12:
        return 0.35, 0.67
    return float(np.median(p01)), float(np.median(p12))


def progress_prior_proba(progress: np.ndarray, thresholds: Tuple[float, float], temperature: float, num_classes: int) -> np.ndarray:
    if num_classes != 3:
        proba = np.zeros((len(progress), num_classes), dtype=np.float64)
        idx = np.clip(np.rint(progress * (num_classes - 1)).astype(int), 0, num_classes - 1)
        proba[np.arange(len(progress)), idx] = 1.0
        return proba
    t1, t2 = thresholds
    temp = max(float(temperature), 1e-4)
    q1 = 1.0 / (1.0 + np.exp(-(progress - t1) / temp))
    q2 = 1.0 / (1.0 + np.exp(-(progress - t2) / temp))
    proba = np.stack([1.0 - q1, q1 - q2, q2], axis=1)
    proba = np.clip(proba, 1e-8, 1.0)
    return proba / proba.sum(axis=1, keepdims=True)


def blend_probabilities(model_proba: np.ndarray, prior_proba: np.ndarray, prior_weight: float) -> np.ndarray:
    weight = float(np.clip(prior_weight, 0.0, 1.0))
    blended = (1.0 - weight) * model_proba + weight * prior_proba
    blended = np.clip(blended, 1e-8, 1.0)
    return blended / blended.sum(axis=1, keepdims=True)


def tune_progress_blend(
    val_model_proba: np.ndarray,
    val_progress: np.ndarray,
    val_y: np.ndarray,
    val_groups: np.ndarray,
    thresholds: Tuple[float, float],
    num_classes: int,
) -> Tuple[float, float, float]:
    if len(val_y) == 0:
        return 0.5, 0.05, float("nan")
    best_weight = 0.0
    best_temp = 0.05
    best_score = -np.inf
    for temp in (0.02, 0.035, 0.05, 0.075, 0.10, 0.15):
        prior = progress_prior_proba(val_progress, thresholds, temperature=temp, num_classes=num_classes)
        for weight in (0.0, 0.20, 0.40, 0.60, 0.80, 1.0):
            blended = blend_probabilities(val_model_proba, prior, prior_weight=weight)
            pred = decode_by_trial(blended, val_groups)
            score = float(f1_score(val_y, pred, average="macro", zero_division=0))
            if score > best_score:
                best_score = score
                best_weight = float(weight)
                best_temp = float(temp)
    return best_weight, best_temp, best_score


def _metric_payload(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, object]:
    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class_recall: List[float] = []
    per_class_precision: List[float] = []
    for c in range(num_classes):
        tp = int(cm[c, c])
        fn = int(cm[c, :].sum()) - tp
        fp = int(cm[:, c].sum()) - tp
        per_class_recall.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        per_class_precision.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": cm.tolist(),
        "per_class_recall": per_class_recall,
        "per_class_precision": per_class_precision,
    }


def _fmt_mean_std(mean: object, std: object) -> str:
    try:
        mean_float = float(mean)
        std_float = float(std)
    except (TypeError, ValueError):
        return "---"
    if np.isnan(mean_float) or np.isnan(std_float):
        return "---"
    return f"${mean_float:.3f} \\pm {std_float:.3f}$"


def _write_model_comparison_table(comparison_df: pd.DataFrame, output_dir: Path, include_progress: bool) -> None:
    metric_specs = [
        ("Raw", "mean_macro_f1", "std_macro_f1"),
        ("Monotonic", "mono_mean_macro_f1", "mono_std_macro_f1"),
    ]
    if include_progress:
        metric_specs.append(("Progress", "progress_mean_macro_f1", "progress_std_macro_f1"))

    lines = [
        r"\begin{tabular}{l" + "c" * len(metric_specs) + "}",
        r"\toprule",
        "Model & " + " & ".join(f"{label} Macro $F_1$" for label, _, _ in metric_specs) + r" \\",
        r"\midrule",
    ]
    for _, row in comparison_df.iterrows():
        display = str(row.get("display_name") or row["model"]).replace("_", r"\_")
        cells = [_fmt_mean_std(row.get(mean_col), row.get(std_col)) for _, mean_col, std_col in metric_specs]
        lines.append(f"{display} & {' & '.join(cells)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (output_dir / "model_comparison.tex").write_text("\n".join(lines))


def run_feature_loso(
    cache_dir: str,
    run_dir: str,
    model_names: Sequence[str],
    seed: int = 42,
) -> Dict[str, object]:
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    x, y, subjects, trials, feature_names = load_feature_table(cache_dir)
    unique_subjects = sorted(set(subjects.tolist()), key=subject_sort_key)
    if len(unique_subjects) < 2:
        raise ValueError("LOSO cross-validation requires at least two subjects.")
    if not model_names:
        raise ValueError("At least one model name is required.")
    unknown_models = [name for name in model_names if name not in AVAILABLE_MODELS]
    if unknown_models:
        raise ValueError(
            f"Unknown model name(s): {', '.join(unknown_models)}. "
            f"Available models: {_available_model_text()}"
        )
    num_classes = int(np.max(y)) + 1
    dataset_summary = {
        "n_windows": int(len(y)),
        "n_features": int(x.shape[1]),
        "subjects": unique_subjects,
        "label_counts": {str(i): int(v) for i, v in enumerate(np.bincount(y, minlength=num_classes))},
    }
    progress_col = feature_names.index("trial_progress") if "trial_progress" in feature_names else None
    (output_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2))

    all_results: Dict[str, object] = {"dataset": dataset_summary, "models": {}}
    comparison_rows: List[Dict[str, object]] = []

    for model_name in model_names:
        model_start = time.time()
        fold_rows: List[Dict[str, object]] = []
        prediction_frames: List[pd.DataFrame] = []
        for fold_index, test_subject in enumerate(unique_subjects):
            train_subjects, val_subjects, _ = split_subjects_for_fold(unique_subjects, test_subject, fold_index, seed=seed)
            train_mask = np.isin(subjects, train_subjects)
            val_mask = np.isin(subjects, val_subjects)
            test_mask = subjects == test_subject
            model = make_model(model_name, seed + fold_index, num_classes=num_classes)
            fold_start = time.time()
            train_y = y[train_mask]
            class_counts = np.bincount(train_y, minlength=num_classes).astype(np.float64)
            class_counts[class_counts == 0] = 1.0
            class_weights = len(train_y) / (num_classes * class_counts)
            sample_weights = class_weights[train_y]
            _fit_model(model, x[train_mask], train_y, sample_weights)
            model_dir = output_dir / "models" / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_dir / f"fold_{fold_index + 1:02d}_{test_subject}.joblib")
            val_proba = _predict_proba(model, x[val_mask], num_classes=num_classes) if np.any(val_mask) else np.empty((0, num_classes), dtype=np.float64)
            val_pred = np.argmax(val_proba, axis=1).astype(np.int64) if np.any(val_mask) else np.empty((0,), dtype=np.int64)
            val_groups = make_group_ids(subjects[val_mask], trials[val_mask]) if np.any(val_mask) else np.empty((0,), dtype=object)
            test_groups = make_group_ids(subjects[test_mask], trials[test_mask])
            val_mono_pred = decode_by_trial(val_proba, val_groups) if np.any(val_mask) else np.empty((0,), dtype=np.int64)
            test_proba = _predict_proba(model, x[test_mask], num_classes=num_classes)
            test_pred = np.argmax(test_proba, axis=1).astype(np.int64)
            test_mono_pred = decode_by_trial(test_proba, test_groups)
            val_macro_f1 = float(f1_score(y[val_mask], val_pred, average="macro", zero_division=0)) if np.any(val_mask) else float("nan")
            val_mono_macro_f1 = float(f1_score(y[val_mask], val_mono_pred, average="macro", zero_division=0)) if np.any(val_mask) else float("nan")
            raw_metrics = _metric_payload(y[test_mask], test_pred, num_classes)
            mono_metrics = _metric_payload(y[test_mask], test_mono_pred, num_classes)
            progress_metrics = None
            progress_weight = None
            progress_temp = None
            progress_val_macro_f1 = None
            progress_thresholds = None
            progress_test_pred = None
            if progress_col is not None:
                train_groups = make_group_ids(subjects[train_mask], trials[train_mask])
                progress_thresholds = fit_progress_thresholds(x[train_mask, progress_col], y[train_mask], train_groups)
                progress_weight, progress_temp, progress_val_macro_f1 = tune_progress_blend(
                    val_model_proba=val_proba,
                    val_progress=x[val_mask, progress_col] if np.any(val_mask) else np.empty((0,), dtype=np.float32),
                    val_y=y[val_mask] if np.any(val_mask) else np.empty((0,), dtype=np.int64),
                    val_groups=val_groups,
                    thresholds=progress_thresholds,
                    num_classes=num_classes,
                )
                test_prior = progress_prior_proba(
                    x[test_mask, progress_col],
                    thresholds=progress_thresholds,
                    temperature=progress_temp,
                    num_classes=num_classes,
                )
                test_blended = blend_probabilities(test_proba, test_prior, prior_weight=progress_weight)
                progress_test_pred = decode_by_trial(test_blended, test_groups)
                progress_metrics = _metric_payload(y[test_mask], progress_test_pred, num_classes)
            row: Dict[str, object] = {
                "model": model_name,
                "fold_index": fold_index,
                "test_subject": test_subject,
                "train_subjects": ",".join(train_subjects),
                "val_subjects": ",".join(val_subjects),
                "test_accuracy": raw_metrics["accuracy"],
                "test_macro_f1": raw_metrics["macro_f1"],
                "test_cohen_kappa": raw_metrics["cohen_kappa"],
                "test_mcc": raw_metrics["mcc"],
                "mono_test_accuracy": mono_metrics["accuracy"],
                "mono_test_macro_f1": mono_metrics["macro_f1"],
                "mono_test_cohen_kappa": mono_metrics["cohen_kappa"],
                "mono_test_mcc": mono_metrics["mcc"],
                "progress_test_accuracy": None if progress_metrics is None else progress_metrics["accuracy"],
                "progress_test_macro_f1": None if progress_metrics is None else progress_metrics["macro_f1"],
                "progress_test_cohen_kappa": None if progress_metrics is None else progress_metrics["cohen_kappa"],
                "progress_test_mcc": None if progress_metrics is None else progress_metrics["mcc"],
                "progress_prior_weight": progress_weight,
                "progress_prior_temperature": progress_temp,
                "progress_threshold_01": None if progress_thresholds is None else progress_thresholds[0],
                "progress_threshold_12": None if progress_thresholds is None else progress_thresholds[1],
                "val_macro_f1": val_macro_f1,
                "val_mono_macro_f1": val_mono_macro_f1,
                "val_progress_macro_f1": progress_val_macro_f1,
                "elapsed_sec": round(time.time() - fold_start, 3),
                "n_train": int(np.sum(train_mask)),
                "n_val": int(np.sum(val_mask)),
                "n_test": int(np.sum(test_mask)),
                "confusion_matrix": raw_metrics["confusion_matrix"],
                "mono_confusion_matrix": mono_metrics["confusion_matrix"],
                "progress_confusion_matrix": None if progress_metrics is None else progress_metrics["confusion_matrix"],
            }
            # Per-class recall and precision for each decoding variant
            for c in range(num_classes):
                row[f"test_recall_c{c}"] = raw_metrics["per_class_recall"][c]
                row[f"test_precision_c{c}"] = raw_metrics["per_class_precision"][c]
                row[f"mono_recall_c{c}"] = mono_metrics["per_class_recall"][c]
                row[f"mono_precision_c{c}"] = mono_metrics["per_class_precision"][c]
                if progress_metrics is not None:
                    row[f"progress_recall_c{c}"] = progress_metrics["per_class_recall"][c]
                    row[f"progress_precision_c{c}"] = progress_metrics["per_class_precision"][c]
                else:
                    row[f"progress_recall_c{c}"] = None
                    row[f"progress_precision_c{c}"] = None
            fold_rows.append(row)
            prediction_data: Dict[str, object] = {
                "model": model_name,
                "test_subject": test_subject,
                "trial_id": trials[test_mask],
                "y_true": y[test_mask],
                "raw_pred": test_pred,
                "mono_pred": test_mono_pred,
                "progress_pred": progress_test_pred if progress_test_pred is not None else test_mono_pred,
            }
            for class_index in range(num_classes):
                prediction_data[f"proba_{class_index}"] = test_proba[:, class_index]
            prediction_frames.append(pd.DataFrame(prediction_data))
            progress_text = (
                f"progress_f1={row['progress_test_macro_f1']:.4f} "
                if progress_metrics is not None
                else ""
            )
            print(
                f"[{model_name}] fold {fold_index + 1:02d}/{len(unique_subjects)} "
                f"{test_subject}: raw_f1={row['test_macro_f1']:.4f} "
                f"mono_f1={row['mono_test_macro_f1']:.4f} "
                f"{progress_text}time={row['elapsed_sec']:.1f}s"
            )

        fold_df = pd.DataFrame(fold_rows)
        fold_df.to_csv(output_dir / f"{model_name}_fold_metrics.csv", index=False)
        predictions_df = pd.concat(prediction_frames, ignore_index=True)

        def _col_mean(col: str) -> float:
            return float(fold_df[col].dropna().mean()) if col in fold_df else float("nan")

        def _col_std(col: str) -> float:
            vals = fold_df[col].dropna() if col in fold_df else pd.Series([], dtype=float)
            return float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")

        summary: Dict[str, object] = {
            "mean_macro_f1": _col_mean("test_macro_f1"),
            "std_macro_f1": _col_std("test_macro_f1"),
            "mean_accuracy": _col_mean("test_accuracy"),
            "std_accuracy": _col_std("test_accuracy"),
            "mean_cohen_kappa": _col_mean("test_cohen_kappa"),
            "mean_mcc": _col_mean("test_mcc"),
            "mono_mean_macro_f1": _col_mean("mono_test_macro_f1"),
            "mono_std_macro_f1": _col_std("mono_test_macro_f1"),
            "mono_mean_accuracy": _col_mean("mono_test_accuracy"),
            "mono_mean_cohen_kappa": _col_mean("mono_test_cohen_kappa"),
            "mono_mean_mcc": _col_mean("mono_test_mcc"),
            "progress_mean_macro_f1": _col_mean("progress_test_macro_f1"),
            "progress_std_macro_f1": _col_std("progress_test_macro_f1"),
            "progress_mean_accuracy": _col_mean("progress_test_accuracy"),
            "progress_mean_cohen_kappa": _col_mean("progress_test_cohen_kappa"),
            "progress_mean_mcc": _col_mean("progress_test_mcc"),
            "total_elapsed_sec": round(time.time() - model_start, 3),
            "folds": fold_rows,
        }
        # Per-class mean recall across folds
        for c in range(num_classes):
            for variant_prefix in ("test", "mono", "progress"):
                col = f"{variant_prefix}_recall_c{c}"
                summary[f"mean_{col}"] = _col_mean(col)

        all_results["models"][model_name] = summary

        # Post-hoc analysis: statistical tests, feature importance, figures
        from .analysis import run_post_analysis
        analysis_out = run_post_analysis(
            fold_df=fold_df,
            predictions_df=predictions_df,
            model_dir=output_dir / "models" / model_name,
            feature_names=feature_names,
            num_classes=num_classes,
            output_dir=output_dir,
            model_name=model_name,
        )
        all_results["models"][model_name]["analysis"] = analysis_out
        comparison_rows.append({
            "model": model_name,
            "display_name": MODEL_DISPLAY_NAMES.get(model_name, model_name),
            "mean_macro_f1": summary["mean_macro_f1"],
            "std_macro_f1": summary["std_macro_f1"],
            "mean_accuracy": summary["mean_accuracy"],
            "std_accuracy": summary["std_accuracy"],
            "mean_cohen_kappa": summary["mean_cohen_kappa"],
            "mean_mcc": summary["mean_mcc"],
            "mono_mean_macro_f1": summary["mono_mean_macro_f1"],
            "mono_std_macro_f1": summary["mono_std_macro_f1"],
            "mono_mean_accuracy": summary["mono_mean_accuracy"],
            "mono_mean_cohen_kappa": summary["mono_mean_cohen_kappa"],
            "mono_mean_mcc": summary["mono_mean_mcc"],
            "progress_mean_macro_f1": summary["progress_mean_macro_f1"],
            "progress_std_macro_f1": summary["progress_std_macro_f1"],
            "progress_mean_accuracy": summary["progress_mean_accuracy"],
            "progress_mean_cohen_kappa": summary["progress_mean_cohen_kappa"],
            "progress_mean_mcc": summary["progress_mean_mcc"],
            "total_elapsed_sec": summary["total_elapsed_sec"],
        })

    sort_col = "progress_mean_macro_f1" if progress_col is not None else "mono_mean_macro_f1"
    comparison_df = pd.DataFrame(comparison_rows).sort_values(sort_col, ascending=False)
    comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
    _write_model_comparison_table(comparison_df, output_dir, include_progress=progress_col is not None)
    all_results["comparison"] = comparison_df.to_dict(orient="records")
    return all_results
