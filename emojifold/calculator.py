"""
Semantic distance calculator for emoji pairs
"""

import asyncio
import httpx
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import logging

from .models import EmojiPair, SemanticDistance, ModelConfig, STANDARD_MODELS

logger = logging.getLogger(__name__)


class SemanticCalculator:
    """Calculate semantic distances between emoji pairs using various embedding models"""
    
    def __init__(self, model_config: Union[str, ModelConfig]):
        if isinstance(model_config, str):
            if model_config not in STANDARD_MODELS:
                raise ValueError(f"Unknown model: {model_config}. Available: {list(STANDARD_MODELS.keys())}")
            model_config = STANDARD_MODELS[model_config]
        
        self.config = model_config
        self.client = None
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model based on configuration"""
        if self.config.model_type == "ollama":
            self.client = httpx.AsyncClient(timeout=30.0)
            logger.info(f"Initialized Ollama client for {self.config.name}")
            
        elif self.config.model_type == "sentence_transformer":
            self.model = SentenceTransformer(self.config.model_path)
            if self.config.device:
                self.model = self.model.to(self.config.device)
            logger.info(f"Loaded SentenceTransformer: {self.config.name}")
            
        elif self.config.model_type == "huggingface":
            # TODO: Implement HuggingFace transformer loading
            raise NotImplementedError("HuggingFace models not yet implemented")
        
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
    
    async def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text input"""
        if self.config.model_type == "ollama":
            return await self._get_ollama_embedding(text)
        elif self.config.model_type == "sentence_transformer":
            return self._get_sentence_transformer_embedding(text)
        else:
            raise NotImplementedError(f"Embedding not implemented for {self.config.model_type}")
    
    async def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for multiple texts (batched for efficiency)"""
        if self.config.model_type == "sentence_transformer":
            # Batch process for efficiency
            embeddings = self.model.encode(
                texts, 
                batch_size=self.config.batch_size,
                show_progress_bar=len(texts) > 100
            )
            return [np.array(emb) for emb in embeddings]
        else:
            # Fallback to individual calls with concurrency limit
            semaphore = asyncio.Semaphore(10)  # Limit concurrent requests
            
            async def get_one(text):
                async with semaphore:
                    return await self.get_embedding(text)
            
            tasks = [get_one(text) for text in texts]
            return await asyncio.gather(*tasks)
    
    async def _get_ollama_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Ollama API"""
        response = await self.client.post(
            "http://localhost:11434/api/embeddings",
            json={
                "model": self.config.model_path,
                "prompt": text
            }
        )
        response.raise_for_status()
        result = response.json()
        return np.array(result["embedding"])
    
    def _get_sentence_transformer_embedding(self, text: str) -> np.ndarray:
        """Get embedding from SentenceTransformer model"""
        embedding = self.model.encode([text], batch_size=1)[0]
        return np.array(embedding)
    
    async def calculate_distance(self, pair: EmojiPair) -> SemanticDistance:
        """Calculate semantic distance for a single emoji pair"""
        # Use emoji + description for richer semantics
        text1 = f"{pair.emoji1} {pair.description1 or ''}".strip()
        text2 = f"{pair.emoji2} {pair.description2 or ''}".strip()
        
        # Get embeddings
        embedding1, embedding2 = await self.get_embeddings([text1, text2])
        
        # Calculate cosine distance and similarity
        distance = cosine(embedding1, embedding2)
        similarity = 1 - distance
        
        return SemanticDistance(
            pair=pair,
            model_name=self.config.name,
            distance=distance,
            similarity=similarity,
            embedding1=embedding1.tolist(),
            embedding2=embedding2.tolist()
        )
    
    async def calculate_distances(self, pairs: List[EmojiPair]) -> List[SemanticDistance]:
        """Calculate semantic distances for multiple pairs (optimized)"""
        logger.info(f"Calculating distances for {len(pairs)} pairs using {self.config.name}")
        
        # Prepare all texts for batch embedding
        texts = []
        for pair in pairs:
            text1 = f"{pair.emoji1} {pair.description1 or ''}".strip()
            text2 = f"{pair.emoji2} {pair.description2 or ''}".strip()
            texts.extend([text1, text2])
        
        # Get all embeddings in batch
        all_embeddings = await self.get_embeddings(texts)
        
        # Calculate distances
        results = []
        for i, pair in enumerate(pairs):
            embedding1 = all_embeddings[i * 2]
            embedding2 = all_embeddings[i * 2 + 1]
            
            distance = cosine(embedding1, embedding2)
            similarity = 1 - distance
            
            results.append(SemanticDistance(
                pair=pair,
                model_name=self.config.name,
                distance=distance,
                similarity=similarity,
                embedding1=embedding1.tolist(),
                embedding2=embedding2.tolist()
            ))
        
        logger.info(f"Completed {len(results)} distance calculations")
        return results
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()


# Utility functions for quick calculations
async def quick_distance(emoji1: str, emoji2: str, model: str = "all_minilm") -> float:
    """Quick semantic distance calculation between two emoji"""
    pair = EmojiPair(emoji1, emoji2)
    
    async with SemanticCalculator(model) as calc:
        result = await calc.calculate_distance(pair)
        return result.distance


def cosine_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate cosine distance between two vectors"""
    return cosine(v1, v2)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    return 1 - cosine(v1, v2)
