"""
EmojiFold: Emoji Semantic Manifold Discovery

Core package for discovering semantic structure in emoji through oppositional pair analysis.
"""

__version__ = "0.1.0"
__author__ = "EmojiFold Team"

from .calculator import SemanticCalculator
from .models import EmojiPair, SemanticDistance, AnalysisResult
from .storage import EmojiDatabase
from .batch import BatchProcessor

__all__ = [
    "SemanticCalculator",
    "EmojiPair", 
    "SemanticDistance",
    "AnalysisResult",
    "EmojiDatabase",
    "BatchProcessor",
]
