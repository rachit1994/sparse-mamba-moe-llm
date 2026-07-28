"""The control-arm baseline model: a plain, correct, boring transformer.

Public entry points: :class:`src.models.tiny_lm.TinyLM` and
:class:`src.models.config.ModelConfig`. See `implementation/06_agent_briefs/B3_baseline_model.md`.
"""

from src.models.config import ModelConfig, PRESET_CONFIGS, analytic_param_count
from src.models.tiny_lm import TinyLM, causal_lm_shift, truncate_ids

__all__ = [
    "ModelConfig",
    "PRESET_CONFIGS",
    "TinyLM",
    "analytic_param_count",
    "causal_lm_shift",
    "truncate_ids",
]
