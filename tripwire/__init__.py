"""TRIPWIRE-LOB: sequential shift alarms, expert-tracking adaptation and loud
degradation for short-horizon limit-order-book forecasting."""

from .config import TripwireConfig
from .features import FeatureExtractor, FEATURE_NAMES, N_FEATURES
from .engine import TripwireEngine

__all__ = ["TripwireConfig", "FeatureExtractor", "FEATURE_NAMES", "N_FEATURES", "TripwireEngine"]
__version__ = "1.0.0"
