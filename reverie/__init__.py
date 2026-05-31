"""Adaptive, curriculum-free reasoning in continuous latent space.

JAX take on Coconut (Hao et al., 2024). See docs/DESIGN.md.
"""

from reverie.latent import ReverieConfig, ReverieModel
from reverie.model import ModelConfig, Transformer

__all__ = ["ModelConfig", "Transformer", "ReverieConfig", "ReverieModel"]
__version__ = "0.1.0"
