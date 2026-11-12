"""Fast sEMG fatigue modeling utilities."""

from .fast_features import FeatureBuildConfig, build_feature_cache, run_feature_loso

__version__ = "0.5.0"

__all__ = [
    "FeatureBuildConfig",
    "build_feature_cache",
    "run_feature_loso",
    "__version__",
]
