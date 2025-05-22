"""
Database storage for EmojiFold results
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from .models import EmojiPair, SemanticDistance, AnalysisResult

logger = logging.getLogger(__name__)


class EmojiDatabase:
    """SQLite database for storing emoji semantic analysis results"""
    
    def __init__(self, db_path: str = "./data/emojifold.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS emoji_pairs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    emoji1 TEXT NOT NULL,
                    emoji2 TEXT NOT NULL,
                    description1 TEXT,
                    description2 TEXT,
                    pair_id TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS semantic_distances (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    distance REAL NOT NULL,
                    similarity REAL NOT NULL,
                    embedding1 TEXT,  -- JSON serialized embedding
                    embedding2 TEXT,  -- JSON serialized embedding
                    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (pair_id) REFERENCES emoji_pairs (pair_id),
                    UNIQUE(pair_id, model_name)
                );
                
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    total_pairs INTEGER NOT NULL,
                    completion_time TIMESTAMP NOT NULL,
                    statistics TEXT,  -- JSON serialized stats
                    metadata TEXT,    -- JSON serialized metadata
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_semantic_distances_model 
                    ON semantic_distances(model_name);
                CREATE INDEX IF NOT EXISTS idx_semantic_distances_distance 
                    ON semantic_distances(distance DESC);
                CREATE INDEX IF NOT EXISTS idx_pairs_emoji 
                    ON emoji_pairs(emoji1, emoji2);
            """)
    
    def store_pair(self, pair: EmojiPair) -> None:
        """Store an emoji pair"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR IGNORE INTO emoji_pairs 
                (emoji1, emoji2, description1, description2, pair_id)
                VALUES (?, ?, ?, ?, ?)
            """, (
                pair.emoji1, pair.emoji2, 
                pair.description1, pair.description2, 
                pair.pair_id
            ))
    
    def store_distance(self, distance: SemanticDistance) -> None:
        """Store a semantic distance result"""
        # First ensure the pair exists
        self.store_pair(distance.pair)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO semantic_distances
                (pair_id, model_name, distance, similarity, embedding1, embedding2, calculated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                distance.pair.pair_id,
                distance.model_name,
                distance.distance,
                distance.similarity,
                json.dumps(distance.embedding1) if distance.embedding1 else None,
                json.dumps(distance.embedding2) if distance.embedding2 else None,
                distance.calculated_at
            ))
    
    def store_distances(self, distances: List[SemanticDistance]) -> None:
        """Store multiple semantic distance results efficiently"""
        logger.info(f"Storing {len(distances)} distance results")
        
        with sqlite3.connect(self.db_path) as conn:
            # Store pairs first
            pair_data = []
            for dist in distances:
                pair = dist.pair
                pair_data.append((
                    pair.emoji1, pair.emoji2,
                    pair.description1, pair.description2,
                    pair.pair_id
                ))
            
            conn.executemany("""
                INSERT OR IGNORE INTO emoji_pairs 
                (emoji1, emoji2, description1, description2, pair_id)
                VALUES (?, ?, ?, ?, ?)
            """, pair_data)
            
            # Store distances
            distance_data = []
            for dist in distances:
                distance_data.append((
                    dist.pair.pair_id,
                    dist.model_name,
                    dist.distance,
                    dist.similarity,
                    json.dumps(dist.embedding1) if dist.embedding1 else None,
                    json.dumps(dist.embedding2) if dist.embedding2 else None,
                    dist.calculated_at
                ))
            
            conn.executemany("""
                INSERT OR REPLACE INTO semantic_distances
                (pair_id, model_name, distance, similarity, embedding1, embedding2, calculated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, distance_data)
    
    def get_strongest_oppositions(
        self, 
        model_name: str, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get strongest semantic oppositions for a model"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT 
                    p.emoji1, p.emoji2, p.description1, p.description2,
                    sd.distance, sd.similarity, sd.calculated_at
                FROM semantic_distances sd
                JOIN emoji_pairs p ON sd.pair_id = p.pair_id
                WHERE sd.model_name = ?
                ORDER BY sd.distance DESC
                LIMIT ?
            """, (model_name, limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_model_statistics(self, model_name: str) -> Dict[str, float]:
        """Get statistics for a model's results"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_pairs,
                    AVG(distance) as mean_distance,
                    MIN(distance) as min_distance,
                    MAX(distance) as max_distance,
                    AVG(similarity) as mean_similarity
                FROM semantic_distances
                WHERE model_name = ?
            """, (model_name,))
            
            row = cursor.fetchone()
            return {
                "total_pairs": row[0],
                "mean_distance": row[1] or 0.0,
                "min_distance": row[2] or 0.0,
                "max_distance": row[3] or 0.0,
                "mean_similarity": row[4] or 0.0,
            }
    
    def get_cross_model_comparison(self, pair_id: str) -> List[Dict[str, Any]]:
        """Get all model results for a specific pair"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT model_name, distance, similarity, calculated_at
                FROM semantic_distances
                WHERE pair_id = ?
                ORDER BY distance DESC
            """, (pair_id,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_processed_pairs(self, model_name: str) -> set:
        """Get set of pair_ids already processed for a model (for resume functionality)"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT pair_id FROM semantic_distances WHERE model_name = ?
            """, (model_name,))
            
            return {row[0] for row in cursor.fetchall()}
    
    def store_analysis_result(self, result: AnalysisResult) -> None:
        """Store complete analysis results"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO analysis_results
                (model_name, total_pairs, completion_time, statistics, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                result.model_name,
                result.total_pairs,
                result.completion_time,
                json.dumps(result.statistics),
                json.dumps(result.metadata)
            ))
    
    def export_results(self, model_name: str, output_path: str) -> None:
        """Export results to JSON file"""
        results = self.get_strongest_oppositions(model_name, limit=10000)
        stats = self.get_model_statistics(model_name)
        
        export_data = {
            "model_name": model_name,
            "statistics": stats,
            "results": results,
            "exported_at": datetime.utcnow().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(results)} results to {output_path}")
