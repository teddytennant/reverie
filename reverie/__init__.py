"""Reverie — adaptive, curriculum-free reasoning in continuous latent space.

A successor to Coconut (Hao et al., 2024) implemented in JAX. See docs/DESIGN.md.
"""

from reverie.model import ModelConfig, Transformer

__all__ = ["ModelConfig", "Transformer"]
__version__ = "0.1.0"
