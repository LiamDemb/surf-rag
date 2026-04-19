import os
from dataclasses import dataclass

@dataclass(frozen=True)
class ScoringConfig:
    local_pred_weight: float = 0.7
    bundle_pred_weight: float = 0.6
    length_penalty: float = 0.05

def get_default_scoring_config() -> ScoringConfig:
    return ScoringConfig(
        local_pred_weight=float(os.getenv("GRAPH_LOCAL_PRED_WEIGHT", "0.7")),
        bundle_pred_weight=float(os.getenv("GRAPH_BUNDLE_PRED_WEIGHT", "0.6")),
        length_penalty=float(os.getenv("GRAPH_LENGTH_PENALTY", "0.05")),
    )

DEFAULT_SCORING_CONFIG = get_default_scoring_config()
