"""Registry mapping method names to SteeringMethod classes."""

from .base import SteeringMethod
from .no_steering import NoSteering
from .random_direction import RandomDirection
from .caa import CAAMeanDiff, CAAContrastive, CAARepE
from .single_feature import SingleFeature
from .topk_features import TopKFeatures
from .convex_optimal import ConvexOptimalSteering
from .qp_optimal import QPOptimalSteering

STEERING_METHODS: dict[str, type] = {
    "no_steering": NoSteering,
    "random_direction": RandomDirection,
    "caa_mean_diff": CAAMeanDiff,
    "caa_contrastive": CAAContrastive,
    "caa_repe": CAARepE,
    "single_feature": SingleFeature,
    "topk_features": TopKFeatures,
    "convex_optimal": ConvexOptimalSteering,
    "qp_optimal": QPOptimalSteering,
}


def get_steering_method(name: str, **kwargs) -> SteeringMethod:
    """Instantiate a steering method by name.

    Args:
        name: Method name (key in STEERING_METHODS).
        **kwargs: Passed to the method constructor.

    Returns:
        An instance of the requested SteeringMethod.

    Raises:
        ValueError: If the method name is not recognized.
    """
    if name not in STEERING_METHODS:
        available = ", ".join(sorted(STEERING_METHODS.keys()))
        raise ValueError(f"Unknown steering method '{name}'. Available: {available}")
    return STEERING_METHODS[name](**kwargs)
