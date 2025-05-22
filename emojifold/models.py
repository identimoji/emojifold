"""
Data models for EmojiFold analysis
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime


@dataclass
class EmojiPair:
    """Represents a pair of emojis for semantic analysis"""
    emoji1: str
    emoji2: str
    description1: Optional[str] = None
    description2: Optional[str] = None
    
    def __post_init__(self):
        # Ensure consistent ordering for caching
        if self.emoji1 > self.emoji2:
            self.emoji1, self.emoji2 = self.emoji2, self.emoji1
            self.description1, self.description2 = self.description2, self.description1
    
    @property
    def pair_id(self) -> str:
        """Unique identifier for this pair"""
        return f"{self.emoji1}_{self.emoji2}"
    
    def __str__(self) -> str:
        return f"{self.emoji1} vs {self.emoji2}"


@dataclass
class SemanticDistance:
    """Results of semantic distance calculation between emoji pair"""
    pair: EmojiPair
    model_name: str
    distance: float
    similarity: float
    embedding1: Optional[List[float]] = None
    embedding2: Optional[List[float]] = None
    calculated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.calculated_at is None:
            self.calculated_at = datetime.utcnow()


@dataclass 
class AnalysisResult:
    """Complete analysis results for a batch of emoji pairs"""
    model_name: str
    total_pairs: int
    completion_time: datetime
    strongest_oppositions: List[SemanticDistance]
    weakest_oppositions: List[SemanticDistance]  
    statistics: Dict[str, float]
    metadata: Dict[str, Any]
    
    @property
    def average_distance(self) -> float:
        return self.statistics.get("mean_distance", 0.0)
    
    @property
    def max_distance(self) -> float:
        return self.statistics.get("max_distance", 0.0)


@dataclass
class ModelConfig:
    """Configuration for embedding models"""
    name: str
    model_type: str  # 'ollama', 'huggingface', 'sentence_transformer'
    model_path: str
    embedding_dim: int
    batch_size: int = 32
    max_seq_length: int = 512
    device: Optional[str] = None
    host: Optional[str] = None  # For Ollama models
    
    # Model-specific parameters
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


# Common model configurations
STANDARD_MODELS = {
    "nomic_embed": ModelConfig(
        name="nomic-embed-text",
        model_type="ollama", 
        model_path="nomic-embed-text",
        embedding_dim=768,
        batch_size=64,
    ),
    "all_minilm": ModelConfig(
        name="all-MiniLM-L6-v2",
        model_type="sentence_transformer",
        model_path="sentence-transformers/all-MiniLM-L6-v2", 
        embedding_dim=384,
        batch_size=128,
    ),
    "math_bert": ModelConfig(
        name="math-bert-base",
        model_type="huggingface",
        model_path="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        embedding_dim=768,
        batch_size=32,
    ),
}
