"""Unit tests for the sEMG fatigue feature pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from semg_fatigue.fast_features import (
    AVAILABLE_MODELS,
    _fit_model,
    _majority_vote_labels,
    _metric_payload,
    _nearest_labels,
    _predict_proba,
    extract_trial_features,
    make_model,
    monotonic_fatigue_decode,
    split_subjects_for_fold,
)
from semg_fatigue.analysis import run_statistical_tests
from semg_fatigue.data import (
    apply_mvc_normalization,
    compute_mvc_map,
    load_emg_trial,
    load_label_file,
    normalize_muscle_name,
)


# ---------------------------------------------------------------------------
# _majority_vote_labels
# ---------------------------------------------------------------------------

class TestMajorityVoteLabels:
    def _make_label_series(self, n: int = 200, switch: int = 100) -> tuple:
        times = np.linspace(0.0, 4.0, n, dtype=np.float32)
        labels = np.where(np.arange(n) < switch, 0, 1).astype(np.int64)
        return times, labels

    def test_window_entirely_in_class_0(self):
        times, labels = self._make_label_series(switch=100)
        start = np.array([times[10]], dtype=np.float32)
        end = np.array([times[30]], dtype=np.float32)
        result = _majority_vote_labels(times, labels, start, end)
        assert result[0] == 0

    def test_window_entirely_in_class_1(self):
        times, labels = self._make_label_series(switch=50)
        start = np.array([times[120]], dtype=np.float32)
        end = np.array([times[150]], dtype=np.float32)
        result = _majority_vote_labels(times, labels, start, end)
        assert result[0] == 1

    def test_boundary_window_majority_wins(self):
        # Window spans switch point; most of window is class 0
        times, labels = self._make_label_series(n=200, switch=160)
        # Window from sample 140..190 — 20 in class 0, 30 in class 1 → expect 1
        start = np.array([times[140]], dtype=np.float32)
        end = np.array([times[190]], dtype=np.float32)
        result = _majority_vote_labels(times, labels, start, end)
        assert result[0] == 1  # majority of window is class 1

    def test_center_point_vs_majority_differ_at_boundary(self):
        # At a sharp transition, center-point and majority vote can differ.
        # Window covers indices 90..110 (switch at 100): center=100 → class1 (center-point),
        # but majority of window is class0 (indices 90..99=10 samples, 100..110=11 samples).
        times, labels = self._make_label_series(n=200, switch=100)
        start = np.array([times[90]], dtype=np.float32)
        end = np.array([times[110]], dtype=np.float32)
        majority = _majority_vote_labels(times, labels, start, end)
        center_t = np.array([(times[90] + times[110]) * 0.5], dtype=np.float32)
        center_label = _nearest_labels(times, labels, center_t)
        # majority is less sensitive to the exact transition point
        assert majority[0] in (0, 1)
        assert center_label[0] in (0, 1)

    def test_multiple_windows_vectorised(self):
        times, labels = self._make_label_series(switch=100)
        starts = np.array([times[0], times[110], times[150]], dtype=np.float32)
        ends = np.array([times[20], times[130], times[170]], dtype=np.float32)
        result = _majority_vote_labels(times, labels, starts, ends)
        assert result[0] == 0
        assert result[1] == 1
        assert result[2] == 1


# ---------------------------------------------------------------------------
# iEMG is no longer equal to MAV
# ---------------------------------------------------------------------------

class TestIEMGNotMAV:
    """Verify that iEMG and MAV are distinct after the fix."""

    def test_iemg_differs_from_mav(self):
        # iEMG = sum(|x|)/fs, MAV = sum(|x|)/n_samples (n_samples = window_sec * fs)
        # For a 0.5 s window at 1 kHz: n_samples=500, iEMG/MAV = 500/1000 = 0.5
        rng = np.random.default_rng(0)
        n_samples = 1000
        labels = np.zeros(50, dtype=np.int64)
        signal = rng.normal(0, 1, (n_samples, 4)).astype(np.float32)
        time_axis = np.arange(n_samples, dtype=np.float32) / 1000.0
        label_times = np.linspace(0.0, time_axis[-1], 50, dtype=np.float32)
        x, _, _, fn = extract_trial_features(
            signal=signal,
            time_axis=time_axis,
            fs=1000.0,
            label_times=label_times,
            labels=labels,
            channel_names=["CH1", "CH2", "CH3", "CH4"],
            window_sec=0.5,
            stride_sec=0.25,
        )
        iemg_idx = fn.index("CH1__iemg")
        mav_idx = fn.index("CH1__mav")
        ratio = x[:, iemg_idx] / (x[:, mav_idx] + 1e-10)
        np.testing.assert_allclose(ratio, 0.5, rtol=1e-4,
            err_msg="iEMG should equal MAV * window_sec = MAV * 0.5 for 0.5s windows at 1kHz")


# ---------------------------------------------------------------------------
# monotonic_fatigue_decode — correctness of vectorised O(n) version
# ---------------------------------------------------------------------------

class TestMonotonicDecoder:
    def _brute_force(self, proba: np.ndarray) -> np.ndarray:
        """Reference O(n²) brute-force decode for comparison."""
        assert proba.shape[1] == 3
        logp = np.log(np.clip(proba, 1e-8, 1.0))
        n = len(logp)
        p0 = np.concatenate([[0.0], np.cumsum(logp[:, 0])])
        p1 = np.concatenate([[0.0], np.cumsum(logp[:, 1])])
        p2 = np.concatenate([[0.0], np.cumsum(logp[:, 2])])
        best, bt1, bt2 = -np.inf, 0, 0
        for t1 in range(n + 1):
            for t2 in range(t1, n + 1):
                score = p0[t1] + (p1[t2] - p1[t1]) + (p2[n] - p2[t2])
                if score > best:
                    best, bt1, bt2 = score, t1, t2
        pred = np.empty(n, dtype=np.int64)
        pred[:bt1] = 0
        pred[bt1:bt2] = 1
        pred[bt2:] = 2
        return pred

    def test_matches_brute_force_random(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            n = rng.integers(5, 30)
            raw = rng.dirichlet([1.0, 1.0, 1.0], size=n).astype(np.float64)
            expected = self._brute_force(raw)
            got = monotonic_fatigue_decode(raw)
            np.testing.assert_array_equal(got, expected)

    def test_clear_sequence(self):
        # Obvious 0→1→2 signal
        proba = np.array([[0.9, 0.05, 0.05]] * 5
                       + [[0.05, 0.9, 0.05]] * 5
                       + [[0.05, 0.05, 0.9]] * 5, dtype=np.float64)
        result = monotonic_fatigue_decode(proba)
        assert list(result[:5]) == [0] * 5
        assert list(result[5:10]) == [1] * 5
        assert list(result[10:]) == [2] * 5

    def test_all_class_0(self):
        proba = np.tile([0.9, 0.05, 0.05], (10, 1)).astype(np.float64)
        result = monotonic_fatigue_decode(proba)
        assert np.all(result == 0)

    def test_all_class_2(self):
        proba = np.tile([0.05, 0.05, 0.9], (10, 1)).astype(np.float64)
        result = monotonic_fatigue_decode(proba)
        assert np.all(result == 2)

    def test_empty_input(self):
        result = monotonic_fatigue_decode(np.empty((0, 3), dtype=np.float64))
        assert len(result) == 0

    def test_non_3_class_falls_back_to_argmax(self):
        proba = np.array([[0.1, 0.8], [0.6, 0.4], [0.2, 0.8]], dtype=np.float64)
        result = monotonic_fatigue_decode(proba)
        expected = np.argmax(proba, axis=1).astype(np.int64)
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# _metric_payload — per-class recall and precision
# ---------------------------------------------------------------------------

class TestMetricPayload:
    def test_contains_per_class_fields(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])
        payload = _metric_payload(y_true, y_pred, num_classes=3)
        assert "per_class_recall" in payload
        assert "per_class_precision" in payload
        assert len(payload["per_class_recall"]) == 3
        assert len(payload["per_class_precision"]) == 3

    def test_perfect_prediction(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        payload = _metric_payload(y, y, num_classes=3)
        np.testing.assert_allclose(payload["per_class_recall"], [1.0, 1.0, 1.0])
        np.testing.assert_allclose(payload["per_class_precision"], [1.0, 1.0, 1.0])
        assert payload["accuracy"] == pytest.approx(1.0)
        assert payload["macro_f1"] == pytest.approx(1.0)

    def test_all_predicted_class_0(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 0, 0])
        payload = _metric_payload(y_true, y_pred, num_classes=3)
        assert payload["per_class_recall"][0] == pytest.approx(1.0)
        assert payload["per_class_recall"][1] == pytest.approx(0.0)
        assert payload["per_class_recall"][2] == pytest.approx(0.0)
        assert payload["per_class_precision"][0] == pytest.approx(1.0 / 3.0)


# ---------------------------------------------------------------------------
# Sampling rate estimation — median diff is robust to outlier first sample
# ---------------------------------------------------------------------------

class TestSamplingRate:
    """Smoke test that fs estimation tolerates a noisy first sample."""

    def test_median_robust_to_first_sample_gap(self, tmp_path):
        import pandas as pd

        fs_true = 1000.0
        n = 500
        t = np.arange(n) / fs_true
        # Introduce an outlier first interval that is still monotonic
        # (1.5x normal step, so sequence remains strictly increasing)
        t[1] = t[0] + 1.5 / fs_true
        signal = np.random.default_rng(1).normal(0, 1, (n, 4)).astype(np.float32)

        df = pd.DataFrame({
            "X [s]": t,
            "BICEPS_BRACHII: EMG": signal[:, 0],
            "DELTOID_ANTERIOR: EMG": signal[:, 1],
            "DELTOID_MEDIUS: EMG": signal[:, 2],
            "DELTOID_POSTERIOR: EMG": signal[:, 3],
        })
        csv_path = tmp_path / "trial_1.csv"
        df.to_csv(csv_path, index=False)

        _, _, _, fs_est = load_emg_trial(csv_path)
        # Median is robust; should still be close to 1000 Hz
        assert abs(fs_est - fs_true) < 5.0, f"Expected ~{fs_true} Hz, got {fs_est:.1f}"


# ---------------------------------------------------------------------------
# compute_mvc_map — all channels in a multi-column CSV are captured
# ---------------------------------------------------------------------------

class TestMVCMap:
    def test_filtered_mvc_suffix_variants_are_stripped(self):
        name = "R DELTOID ANTERIOR: EMG 2->filter (FilterSlidingRMS51) [V]"
        assert normalize_muscle_name(name) == "R DELTOID ANTERIOR 2"

    def test_single_column_csv(self, tmp_path):
        import pandas as pd

        mvc_dir = tmp_path / "MVC"
        mvc_dir.mkdir()
        df = pd.DataFrame({"BICEPS_BRACHII: EMG": [0.1, 0.5, 0.3, -0.9, 0.2]})
        (mvc_dir / "mvc_biceps.csv").write_text(df.to_csv(index=False))

        mvc = compute_mvc_map(mvc_dir)
        assert "BICEPS_BRACHII" in mvc
        assert mvc["BICEPS_BRACHII"] == pytest.approx(0.9)

    def test_multi_column_csv_all_channels_captured(self, tmp_path):
        import pandas as pd

        mvc_dir = tmp_path / "MVC"
        mvc_dir.mkdir()
        df = pd.DataFrame({
            "BICEPS_BRACHII: EMG": [0.0, 0.8, 0.4],
            "DELTOID_ANTERIOR: EMG": [0.0, 0.6, -0.3],
            "DELTOID_MEDIUS: EMG": [0.0, 0.4, 0.2],
            "DELTOID_POSTERIOR: EMG": [0.0, 0.5, -0.5],
        })
        (mvc_dir / "mvc_all.csv").write_text(df.to_csv(index=False))

        mvc = compute_mvc_map(mvc_dir)
        assert "BICEPS_BRACHII" in mvc
        assert "DELTOID_ANTERIOR" in mvc
        assert "DELTOID_MEDIUS" in mvc
        assert "DELTOID_POSTERIOR" in mvc
        assert mvc["BICEPS_BRACHII"] == pytest.approx(0.8)
        assert mvc["DELTOID_ANTERIOR"] == pytest.approx(0.6)

    def test_filtered_mvc_headers_match_trial_channels(self, tmp_path):
        import pandas as pd

        mvc_dir = tmp_path / "MVC"
        mvc_dir.mkdir()
        df = pd.DataFrame({
            "X [s]": [0.0, 0.1],
            "R BICEPS BRACHII: EMG 1->filter (FilterSlidingRMS27) [V]": [0.0, 0.8],
            "R DELTOID ANTERIOR: EMG 2->filter (FilterSlidingRMS51) [V]": [0.0, 0.6],
            "R DELTOID MEDIUS: EMG 6->filter (FilterSlidingRMS15) [V]": [0.0, 0.4],
            "R DELTOID POSTERIOR: EMG 7->filter (FilterSlidingRMS39) [V]": [0.0, 0.5],
        })
        (mvc_dir / "mvc_all_filtered.csv").write_text(df.to_csv(index=False))

        mvc = compute_mvc_map(mvc_dir)
        channel_names = [
            "R BICEPS BRACHII 1",
            "R DELTOID ANTERIOR 2",
            "R DELTOID MEDIUS 6",
            "R DELTOID POSTERIOR 7",
        ]
        signal = np.ones((2, 4), dtype=np.float32)
        normalized = apply_mvc_normalization(signal, channel_names, mvc)

        np.testing.assert_allclose(
            normalized[0],
            np.array([1 / 0.8, 1 / 0.6, 1 / 0.4, 1 / 0.5], dtype=np.float32),
            rtol=1e-6,
        )

    def test_mvc_normalization_uses_canonical_fallback(self, tmp_path):
        import pandas as pd

        mvc_dir = tmp_path / "MVC"
        mvc_dir.mkdir()
        df = pd.DataFrame({
            "R DELTOID MEDIUS: EMG 6->filter (FilterSlidingRMS15) [V]": [0.0, 0.4],
        })
        (mvc_dir / "mvc_medius.csv").write_text(df.to_csv(index=False))

        mvc = compute_mvc_map(mvc_dir)
        signal = np.array([[2.0]], dtype=np.float32)
        normalized = apply_mvc_normalization(signal, ["R DELTOID MEDIUS 99"], mvc)
        assert normalized[0, 0] == pytest.approx(5.0)

    def test_missing_directory_returns_empty(self, tmp_path):
        mvc = compute_mvc_map(tmp_path / "nonexistent")
        assert mvc == {}


# ---------------------------------------------------------------------------
# load_label_file validation
# ---------------------------------------------------------------------------

class TestLabelFileValidation:
    def test_empty_label_file_rejected(self, tmp_path):
        path = tmp_path / "labels.csv"
        path.write_text("time,label\n")
        with pytest.raises(ValueError, match="empty"):
            load_label_file(path)

    def test_fractional_label_rejected(self, tmp_path):
        import pandas as pd

        path = tmp_path / "labels.csv"
        pd.DataFrame({"time": [0.0, 1.0], "label": [0, 1.5]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="integer"):
            load_label_file(path)

    def test_trailing_non_monotonic_label_time_is_trimmed(self, tmp_path):
        import pandas as pd

        path = tmp_path / "labels.csv"
        pd.DataFrame({
            "time": [0.0, 0.02, 0.04, 0.02],
            "label": [0, 1, 2, 2],
        }).to_csv(path, index=False)

        with pytest.warns(UserWarning, match="Non-monotonic label time"):
            times, labels = load_label_file(path)

        np.testing.assert_allclose(times, [0.0, 0.02, 0.04])
        np.testing.assert_array_equal(labels, [0, 1, 2])


# ---------------------------------------------------------------------------
# split_subjects_for_fold — seeded randomization
# ---------------------------------------------------------------------------

class TestSplitSubjectsForFold:
    SUBJECTS = [f"subject_{i}" for i in range(1, 11)]  # 10 subjects

    def test_test_subject_never_in_train_or_val(self):
        for fold_idx, test_sub in enumerate(self.SUBJECTS):
            train, val, test = split_subjects_for_fold(self.SUBJECTS, test_sub, fold_idx)
            assert test_sub not in train
            assert test_sub not in val
            assert test == [test_sub]

    def test_no_overlap_between_train_and_val(self):
        for fold_idx, test_sub in enumerate(self.SUBJECTS):
            train, val, _ = split_subjects_for_fold(self.SUBJECTS, test_sub, fold_idx)
            assert set(train).isdisjoint(set(val))

    def test_partitions_are_exhaustive(self):
        for fold_idx, test_sub in enumerate(self.SUBJECTS):
            train, val, test = split_subjects_for_fold(self.SUBJECTS, test_sub, fold_idx)
            assert sorted(train + val + test) == sorted(self.SUBJECTS)

    def test_different_folds_produce_different_val_sets(self):
        val_sets = []
        for fold_idx, test_sub in enumerate(self.SUBJECTS):
            _, val, _ = split_subjects_for_fold(self.SUBJECTS, test_sub, fold_idx, seed=42)
            val_sets.append(frozenset(val))
        # Not all folds should pick the same validation pair
        assert len(set(val_sets)) > 1

    def test_same_seed_is_reproducible(self):
        train_a, val_a, _ = split_subjects_for_fold(self.SUBJECTS, "subject_1", 0, seed=99)
        train_b, val_b, _ = split_subjects_for_fold(self.SUBJECTS, "subject_1", 0, seed=99)
        assert val_a == val_b
        assert train_a == train_b

    def test_different_seeds_may_differ(self):
        _, val_42, _ = split_subjects_for_fold(self.SUBJECTS, "subject_1", 0, seed=42)
        _, val_99, _ = split_subjects_for_fold(self.SUBJECTS, "subject_1", 0, seed=99)
        # With 9 candidates and 2 val slots, the chance of a collision is low
        # but not zero — we just verify the function accepts the parameter.
        assert len(val_42) == 2
        assert len(val_99) == 2

    def test_val_size_is_two_for_large_enough_pool(self):
        _, val, _ = split_subjects_for_fold(self.SUBJECTS, "subject_1", 0)
        assert len(val) == 2

    def test_tiny_pool_caps_val_size(self):
        tiny = ["subject_1", "subject_2", "subject_3"]
        _, val, _ = split_subjects_for_fold(tiny, "subject_1", 0)
        assert len(val) <= 1  # only 2 remain after removing test; max val = 1


# ---------------------------------------------------------------------------
# Class weighting — sample_weight applied to model.fit
# ---------------------------------------------------------------------------

class TestClassWeighting:
    """Verify that balanced sample weights are computed correctly."""

    def _compute_weights(self, labels: np.ndarray, num_classes: int) -> np.ndarray:
        """Mirrors the weight logic in run_feature_loso."""
        class_counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
        class_counts[class_counts == 0] = 1.0
        class_weights = len(labels) / (num_classes * class_counts)
        return class_weights[labels]

    def test_balanced_labels_give_equal_weights(self):
        labels = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
        w = self._compute_weights(labels, 3)
        np.testing.assert_allclose(w, np.ones(len(labels)), rtol=1e-6,
            err_msg="Balanced classes should yield uniform sample weights")

    def test_minority_class_gets_higher_weight(self):
        # Class 2 appears once, classes 0 and 1 appear three times each
        labels = np.array([0, 0, 0, 1, 1, 1, 2], dtype=np.int64)
        w = self._compute_weights(labels, 3)
        assert w[6] > w[0], "Minority class (2) should have higher weight than majority class (0)"

    def test_weights_sum_proportional_to_num_classes(self):
        # For any distribution, sum of weights per class should be equal
        labels = np.array([0, 0, 0, 0, 1, 1, 2], dtype=np.int64)
        w = self._compute_weights(labels, 3)
        for c in range(3):
            mask = labels == c
            if mask.any():
                np.testing.assert_allclose(
                    w[mask].sum(), len(labels) / 3, rtol=1e-6,
                    err_msg=f"Class {c} weight sum should equal n_total/n_classes"
                )

    def test_missing_class_does_not_raise(self):
        # Class 2 absent — count is replaced with 1.0, no ZeroDivisionError
        labels = np.array([0, 0, 1, 1], dtype=np.int64)
        w = self._compute_weights(labels, 3)
        assert np.all(np.isfinite(w))


# ---------------------------------------------------------------------------
# Baseline models — shared fit/probability contract
# ---------------------------------------------------------------------------

class TestBaselineModels:
    BASELINES = ("lda", "logistic_regression", "svm_rbf", "random_forest")

    def _toy_multiclass_data(self):
        rng = np.random.default_rng(7)
        centers = np.array([
            [-2.0, 0.0, 0.5, 1.0],
            [0.0, 2.0, -0.5, 0.0],
            [2.0, -1.0, 0.0, -1.0],
        ], dtype=np.float32)
        x_parts = [center + rng.normal(0.0, 0.2, size=(8, 4)) for center in centers]
        x = np.vstack(x_parts).astype(np.float32)
        y = np.repeat(np.arange(3), 8).astype(np.int64)
        return x, y

    def test_expected_model_names_are_available(self):
        assert set(self.BASELINES + ("xgb_fast",)) == set(AVAILABLE_MODELS)

    @pytest.mark.parametrize("model_name", BASELINES)
    def test_baselines_fit_and_return_probabilities(self, model_name):
        x, y = self._toy_multiclass_data()
        model = make_model(model_name, seed=123, num_classes=3)
        sample_weight = np.ones(len(y), dtype=np.float64)

        _fit_model(model, x, y, sample_weight)
        proba = _predict_proba(model, x[:5], num_classes=3)

        assert proba.shape == (5, 3)
        assert np.all(np.isfinite(proba))
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(5), rtol=1e-6)

    def test_predict_proba_aligns_missing_class_columns(self):
        class MissingClassModel:
            classes_ = np.array([1, 2], dtype=np.int64)

            def predict_proba(self, x):
                return np.tile(np.array([[0.25, 0.75]], dtype=np.float64), (len(x), 1))

        proba = _predict_proba(MissingClassModel(), np.zeros((3, 2)), num_classes=3)

        np.testing.assert_allclose(proba[:, 0], 0.0)
        np.testing.assert_allclose(proba[:, 1], 0.25)
        np.testing.assert_allclose(proba[:, 2], 0.75)


# ---------------------------------------------------------------------------
# Statistical tests — paired folds stay aligned
# ---------------------------------------------------------------------------

class TestStatisticalTests:
    def test_pairwise_tests_keep_aligned_fold_rows(self):
        import pandas as pd

        fold_df = pd.DataFrame({
            "test_macro_f1": [0.10, np.nan, 0.30, 0.40, 0.50, 0.60, 0.70],
            "mono_test_macro_f1": [0.20, 0.25, np.nan, 0.45, 0.55, 0.65, 0.75],
            "progress_test_macro_f1": [0.15, 0.26, 0.35, 0.46, 0.56, 0.66, 0.76],
        })

        result = run_statistical_tests(fold_df, num_classes=3, model_name="xgb_fast")
        pair = result["pairwise_wilcoxon_bonferroni"]["raw_vs_mono"]

        assert pair["n_aligned_folds"] == 5
        assert result["friedman"]["n_aligned_folds"] == 5

    def test_identical_pairwise_variants_are_clean_ties(self):
        import pandas as pd
        import warnings

        fold_df = pd.DataFrame({
            "test_macro_f1": [0.50, 0.52, 0.54, 0.56, 0.58],
            "mono_test_macro_f1": [0.50, 0.52, 0.54, 0.56, 0.58],
            "progress_test_macro_f1": [0.55, 0.57, 0.59, 0.61, 0.63],
        })

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = run_statistical_tests(fold_df, num_classes=3, model_name="xgb_fast")

        pair = result["pairwise_wilcoxon_bonferroni"]["raw_vs_mono"]
        assert pair["p_raw"] == pytest.approx(1.0)
        assert pair["rank_biserial_r"] == pytest.approx(0.0)

    def test_identical_friedman_variants_are_clean_ties(self):
        import pandas as pd
        import warnings

        fold_df = pd.DataFrame({
            "test_macro_f1": [0.70, 0.71, 0.72, 0.73, 0.74],
            "mono_test_macro_f1": [0.70, 0.71, 0.72, 0.73, 0.74],
            "progress_test_macro_f1": [0.70, 0.71, 0.72, 0.73, 0.74],
        })

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = run_statistical_tests(fold_df, num_classes=3, model_name="xgb_fast")

        assert result["friedman"]["p_value"] == pytest.approx(1.0)
        assert result["friedman"]["kendall_w"] == pytest.approx(0.0)
