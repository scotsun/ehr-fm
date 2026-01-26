"""Baseline models for comparison with HAT."""

from src.baselines.core_behrt import BEHRT, BEHRTForSequenceClassification
from src.baselines.heart import HEART, HEARTConfig, HEARTForSequenceClassification
from src.baselines.gt_behrt import (
    GTBEHRT,
    GTBEHRTConfig,
    GTBEHRTForPretraining,
    GTBEHRTForSequenceClassification,
    create_gtbehrt_config,
)
from src.baselines.metric import topk_accuracy, recall_at_k, ndcg_at_k

__all__ = [
    # CORE-BEHRT
    "BEHRT",
    "BEHRTForSequenceClassification",
    # HEART
    "HEART",
    "HEARTConfig",
    "HEARTForSequenceClassification",
    # GT-BEHRT
    "GTBEHRT",
    "GTBEHRTConfig",
    "GTBEHRTForPretraining",
    "GTBEHRTForSequenceClassification",
    "create_gtbehrt_config",
    # Metrics
    "topk_accuracy",
    "recall_at_k",
    "ndcg_at_k",
]
