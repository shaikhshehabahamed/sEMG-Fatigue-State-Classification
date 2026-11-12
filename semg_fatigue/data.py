from __future__ import annotations

import warnings
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.signal import butter, filtfilt, iirnotch

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


CANONICAL_CHANNEL_ORDER = [
    "BICEPS_BRACHII",
    "DELTOID_ANTERIOR",
    "DELTOID_MEDIUS",
    "DELTOID_POSTERIOR",
]
_FILTER_SUFFIX_RE = re.compile(r"\s*->\s*FILTER\s*\([^)]*\)", re.IGNORECASE)
_UNIT_SUFFIX_RE = re.compile(r"\s*\[[^\]]+\]")


def subject_sort_key(subject_id: str) -> Tuple[int, str]:
    try:
        return int(subject_id.split("_")[-1]), subject_id
    except Exception:
        return 10**9, subject_id


def butter_bandpass(signal: np.ndarray, fs: float, lowcut: float, highcut: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, signal, axis=0)


def notch_filter(signal: np.ndarray, fs: float, notch_hz: float = 50.0, quality_factor: float = 30.0) -> np.ndarray:
    b, a = iirnotch(notch_hz / (0.5 * fs), quality_factor)
    return filtfilt(b, a, signal, axis=0)


def preprocess_signal(
    signal: np.ndarray,
    fs: float,
    apply_filters: bool,
    bandpass_low: float,
    bandpass_high: float,
    notch_hz: Optional[float],
) -> np.ndarray:
    if fs <= 0:
        raise ValueError(f"Sampling rate must be positive, got {fs}")
    out = signal.astype(np.float32, copy=False)
    if not apply_filters:
        return out
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required for filtering. Install scipy or use --disable-filters.")
    if bandpass_low <= 0 or bandpass_high <= 0 or bandpass_low >= bandpass_high:
        raise ValueError(
            f"Invalid bandpass range: low={bandpass_low}, high={bandpass_high}"
        )
    nyq = 0.5 * fs
    highcut = float(bandpass_high)
    if highcut >= nyq:
        adjusted = nyq * 0.99
        if adjusted <= bandpass_low:
            raise ValueError(
                f"Bandpass high cutoff {bandpass_high} Hz is not valid for fs={fs} Hz"
            )
        warnings.warn(
            f"Bandpass high cutoff {bandpass_high} Hz is at/above Nyquist for fs={fs} Hz; "
            f"using {adjusted:.3f} Hz instead."
        )
        highcut = adjusted
    out = butter_bandpass(out, fs=fs, lowcut=bandpass_low, highcut=highcut)
    if notch_hz is not None and notch_hz > 0:
        if notch_hz >= nyq:
            warnings.warn(
                f"Skipping notch filter at {notch_hz} Hz because it is at/above Nyquist for fs={fs} Hz."
            )
        else:
            out = notch_filter(out, fs=fs, notch_hz=notch_hz)
    return out.astype(np.float32, copy=False)


def normalize_muscle_name(text: str) -> str:
    text = str(text).upper()
    text = text.replace(": EMG", " ")
    text = _FILTER_SUFFIX_RE.sub(" ", text)
    text = _UNIT_SUFFIX_RE.sub(" ", text)
    return " ".join(text.split()).strip()


def canonical_channel_name(channel_name: str) -> Optional[str]:
    tokens = [tok for tok in channel_name.upper().replace("_", " ").split() if not tok.isdigit()]
    if tokens and tokens[0] in {"L", "R", "LEFT", "RIGHT"}:
        tokens = tokens[1:]
    text = " ".join(tokens)
    if "BICEPS" in text:
        return "BICEPS_BRACHII"
    if "DELTOID" in text and "ANTERIOR" in text:
        return "DELTOID_ANTERIOR"
    if "DELTOID" in text and ("MEDIUS" in text or "MIDDLE" in text):
        return "DELTOID_MEDIUS"
    if "DELTOID" in text and "POSTERIOR" in text:
        return "DELTOID_POSTERIOR"
    return None


def canonicalize_channel_order(signal: np.ndarray, channel_names: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    by_name: Dict[str, int] = {}
    for idx, channel_name in enumerate(channel_names):
        canonical = canonical_channel_name(channel_name)
        if canonical is not None and canonical not in by_name:
            by_name[canonical] = idx

    missing = [name for name in CANONICAL_CHANNEL_ORDER if name not in by_name]
    if missing:
        warnings.warn(
            "Could not canonicalize all expected EMG channels "
            f"(missing={missing}, found={list(channel_names)}). Keeping original order."
        )
        return signal, list(channel_names)

    order = [by_name[name] for name in CANONICAL_CHANNEL_ORDER]
    return signal[:, order], list(CANONICAL_CHANNEL_ORDER)


def extract_time_and_signal_columns(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    time_candidates = [c for c in df.columns if c.startswith("X [s]")]
    if not time_candidates:
        raise ValueError("No time column starting with 'X [s]' was found in EMG CSV.")

    signal_cols = [c for c in df.columns if (": EMG" in c) and ("->filter" not in c.lower())]
    if not signal_cols:
        raise ValueError("No raw EMG columns were found in trial file.")

    time = df[time_candidates[0]].to_numpy(dtype=np.float32)
    signal = df[signal_cols].to_numpy(dtype=np.float32)
    channel_names = [normalize_muscle_name(c) for c in signal_cols]
    return time, signal, channel_names


def load_emg_trial(emg_csv: Path) -> Tuple[np.ndarray, np.ndarray, List[str], float]:
    df = pd.read_csv(emg_csv)
    time, signal, channel_names = extract_time_and_signal_columns(df)
    if len(time) == 0:
        raise ValueError(f"No samples found in {emg_csv}")
    if not np.all(np.isfinite(time)):
        raise ValueError(f"Non-finite time values found in {emg_csv}")

    bad_steps = np.flatnonzero(np.diff(time) <= 0)
    if len(bad_steps):
        cut_idx = int(bad_steps[0] + 1)
        warnings.warn(f"Non-monotonic time detected in {emg_csv}; trimming from sample {cut_idx} onward.")
        time = time[:cut_idx]
        signal = signal[:cut_idx]

    if len(time) < 2:
        raise ValueError(f"Not enough samples in {emg_csv}")
    diffs = np.diff(time.astype(np.float64))
    dt = float(np.median(diffs))
    if dt <= 0:
        raise ValueError(f"Non-positive median sampling interval in {emg_csv}")
    return time, signal, channel_names, 1.0 / dt


def load_label_file(label_csv: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(label_csv)
    if not {"time", "label"}.issubset(df.columns):
        raise ValueError(f"Expected columns ['time', 'label'] in {label_csv}")
    if df.empty:
        raise ValueError(f"Label file {label_csv} is empty")
    times = df["time"].to_numpy(dtype=np.float32)
    if not np.all(np.isfinite(times)):
        raise ValueError(f"Non-finite label times found in {label_csv}")
    bad_steps = np.flatnonzero(np.diff(times) <= 0) if len(times) > 1 else np.empty(0, dtype=np.int64)
    if len(bad_steps):
        cut_idx = int(bad_steps[0] + 1)
        warnings.warn(f"Non-monotonic label time detected in {label_csv}; trimming from row {cut_idx} onward.")
        df = df.iloc[:cut_idx].copy()
        times = times[:cut_idx]
    label_values = pd.to_numeric(df["label"], errors="coerce")
    if label_values.isna().any():
        raise ValueError(f"Non-numeric labels found in {label_csv}")
    label_array = label_values.to_numpy(dtype=np.float64)
    if not np.all(np.isclose(label_array, np.rint(label_array))):
        raise ValueError(f"Labels in {label_csv} must be integer class IDs")
    labels = np.rint(label_array).astype(np.int64)
    if np.any(labels < 0):
        raise ValueError(f"Labels in {label_csv} must be non-negative")
    return times, labels


def resolve_label_csv(subject_label_dir: Path, trial_id: str) -> Optional[Path]:
    exact = subject_label_dir / f"{trial_id}.csv"
    if exact.exists():
        return exact
    if not subject_label_dir.exists():
        return None

    target = f"{trial_id}.csv".lower()
    matches = [p for p in subject_label_dir.glob("*.csv") if p.name.lower() == target]
    return sorted(matches)[0] if matches else None


def compute_mvc_map(mvc_dir: Path) -> Dict[str, float]:
    mvc_map: Dict[str, float] = {}
    if not mvc_dir.exists():
        return mvc_map

    def remember_peak(key: Optional[str], peak_value: float) -> None:
        if not key or not np.isfinite(peak_value) or peak_value <= 0:
            return
        previous = mvc_map.get(key)
        if previous is None or peak_value > previous:
            mvc_map[key] = peak_value

    for csv_path in sorted(mvc_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        filtered_cols = [c for c in df.columns if "->filter" in c.lower()]
        raw_cols = [c for c in df.columns if (": EMG" in c) and ("->filter" not in c.lower())]
        cols = filtered_cols or raw_cols
        if not cols:
            continue

        for col in cols:
            values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32)
            if not np.any(np.isfinite(values)):
                continue
            peak = float(np.nanmax(np.abs(values)))
            normalized = normalize_muscle_name(col)
            remember_peak(normalized, peak)
            remember_peak(canonical_channel_name(normalized), peak)
    return mvc_map


def apply_mvc_normalization(signal: np.ndarray, channel_names: Sequence[str], mvc_map: Dict[str, float]) -> np.ndarray:
    out = signal.copy()
    for i, name in enumerate(channel_names):
        peak = mvc_map.get(name)
        if peak is None:
            peak = mvc_map.get(canonical_channel_name(name) or "")
        if peak is not None and peak > 0:
            out[:, i] = out[:, i] / peak
    return out
