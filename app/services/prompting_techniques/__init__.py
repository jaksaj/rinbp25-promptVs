"""
Prompting techniques package - Implements various prompting techniques for LLMs.
"""

from .cot_simple import apply_cot_simple
from .cot_reasoning import apply_cot_reasoning
from .few_shot import apply_few_shot
from .self_consistency import apply_self_consistency

__all__ = ['apply_cot_simple', 'apply_cot_reasoning', 'apply_few_shot', 'apply_self_consistency']
