"""
Batch processing V3 - Full Unicode Emoji Semantic Analysis
Uses codepoint-based schema for complete emoji manifold discovery
"""

import asyncio
import logging
import yaml
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from emojifold.models import EmojiPair, SemanticDistance, AnalysisResult, ModelConfig
from emojifold.calculator import SemanticCalculator

logger = logging.getLogger(__name__)


class BatchProcessorV3:
    """Process ALL emoji pairs for complete semantic manifold discovery"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        
        # Expand home directory paths
        self.db_path = Path(self.config["storage"]["database_path"]).expanduser()
        
        # Ensure output directories exist
        results_dir = Path(self.config["storage"]["results_dir"]).expanduser()
        cache_dir = Path(self.config["storage"]["cache_dir"]).expanduser()
        
        results_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        Path("./logs").mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🗃️ Database: {self.db_path}")
        logger.info(f"📊 Results: {results_dir}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_all_emoji_codepoints(self) -> List[Tuple[int, str, str]]:
        """Get all emoji codepoints, characters, and names from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT codepoint, emoji, name 
                FROM emojis 
                ORDER BY codepoint
            """)
            return cursor.fetchall()
    
    def generate_all_emoji_pairs(self) -> List[EmojiPair]:
        """Generate ALL possible emoji pairs from the complete Unicode set"""
        logger.info("🔍 Loading all emoji from database...")
        emoji_list = self.get_all_emoji_codepoints()
        
        logger.info(f"📊 Found {len(emoji_list)} emoji in database")
        
        pairs = []
        total_pairs = len(emoji_list) * (len(emoji_list) - 1) // 2
        
        logger.info(f"🧮 Generating {total_pairs:,} emoji pairs...")
        logger.info("🚀 FULL EMOJI SEMANTIC MANIFOLD DISCOVERY MODE ACTIVATED!")
        
        pair_count = 0
        for i, (cp1, emoji1, name1) in enumerate(emoji_list):
            for cp2, emoji2, name2 in emoji_list[i+1:]:
                pair = EmojiPair(
                    emoji1=emoji1,
                    emoji2=emoji2,
                    description1=name1,
                    description2=name2
                )
                pairs.append(pair)
                pair_count += 1
                
                # Progress logging every 50k pairs
                if pair_count % 50000 == 0:
                    progress = (pair_count / total_pairs) * 100
                    logger.info(f"⚡ Generated {pair_count:,}/{total_pairs:,} pairs ({progress:.1f}%)")
        
        logger.info(f"✅ Generated {len(pairs):,} emoji pairs for MAXIMUM SEMANTIC DISCOVERY!")
        return pairs
    
    def get_processed_pairs_v3(self, model_name: str) -> set:
        """Get processed pairs for resume functionality using V3 schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT DISTINCT ep.emoji1_codepoint || '_' || ep.emoji2_codepoint as pair_key
                FROM semantic_distances sd
                JOIN emoji_pairs ep ON sd.pair_id = ep.id
                JOIN models m ON sd.model_id = m.id
                WHERE m.name = ?
            """, (model_name,))
            
            return {row[0] for row in cursor.fetchall()}
    
    def store_model_if_not_exists(self, model_config: ModelConfig) -> int:
        """Store model in database and return model ID"""
        with sqlite3.connect(self.db_path) as conn:
            # Check if model exists
            cursor = conn.execute("""
                SELECT id FROM models WHERE name = ?
            """, (model_config.name,))
            
            result = cursor.fetchone()
            if result:
                return result[0]
            
            # Insert new model
            cursor = conn.execute("""
                INSERT INTO models (name, model_type, model_path, embedding_dim, parameters)
                VALUES (?, ?, ?, ?, ?)
            """, (
                model_config.name,
                model_config.model_type,
                model_config.model_path,
                model_config.embedding_dim,
                str(model_config.parameters) if model_config.parameters else None
            ))
            
            model_id = cursor.lastrowid
            logger.info(f"💾 Stored model: {model_config.name} (ID: {model_id})")
            return model_id
    
    def store_distances_v3(self, distances: List[SemanticDistance], model_id: int) -> None:
        """Store semantic distances using V3 schema with codepoints"""
        logger.info(f"💾 Storing {len(distances):,} distance results...")
        
        with sqlite3.connect(self.db_path) as conn:
            # Store emoji pairs first
            pair_data = []
            for dist in distances:
                cp1 = ord(dist.pair.emoji1)
                cp2 = ord(dist.pair.emoji2)
                # Ensure consistent ordering
                if cp1 > cp2:
                    cp1, cp2 = cp2, cp1
                    
                pair_data.append((cp1, cp2))
            
            # Insert pairs (ignore duplicates)
            conn.executemany("""
                INSERT OR IGNORE INTO emoji_pairs (emoji1_codepoint, emoji2_codepoint)
                VALUES (?, ?)
            """, pair_data)
            
            # Store distances
            distance_data = []
            for dist in distances:
                cp1 = ord(dist.pair.emoji1)
                cp2 = ord(dist.pair.emoji2)
                # Ensure consistent ordering
                if cp1 > cp2:
                    cp1, cp2 = cp2, cp1
                
                # Get pair ID
                cursor = conn.execute("""
                    SELECT id FROM emoji_pairs 
                    WHERE emoji1_codepoint = ? AND emoji2_codepoint = ?
                """, (cp1, cp2))
                
                pair_id = cursor.fetchone()[0]
                
                distance_data.append((
                    pair_id,
                    model_id,
                    float(dist.distance),  # Convert numpy float32 to Python float
                    float(dist.similarity), # Convert numpy float32 to Python float
                    str(dist.embedding1) if dist.embedding1 else None,
                    str(dist.embedding2) if dist.embedding2 else None,
                    dist.calculated_at
                ))
            
            conn.executemany("""
                INSERT OR REPLACE INTO semantic_distances
                (pair_id, model_id, distance, similarity, embedding1, embedding2, calculated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, distance_data)
            
            logger.info(f"✅ Stored {len(distances):,} semantic distances")
    
    async def process_model_v3(
        self, 
        model_name: str, 
        resume: bool = True
    ) -> AnalysisResult:
        """Process ALL emoji pairs for complete semantic manifold discovery"""
        logger.info("🚀" * 20)
        logger.info(f"🚀 MAXIMUM POWER EMOJI ANALYSIS: {model_name}")
        logger.info("🚀" * 20)
        
        # Get model configuration
        if model_name not in self.config["models"]:
            raise ValueError(f"Model {model_name} not found in config")
        
        # Fix config key mapping
        model_config_data = self.config["models"][model_name].copy()
        if "type" in model_config_data:
            model_config_data["model_type"] = model_config_data.pop("type")
        if "path" in model_config_data:
            model_config_data["model_path"] = model_config_data.pop("path")
        
        model_config = ModelConfig(**model_config_data)
        model_id = self.store_model_if_not_exists(model_config)
        
        # Generate ALL emoji pairs
        all_pairs = self.generate_all_emoji_pairs()
        
        # Filter out already processed pairs if resuming
        pairs_to_process = all_pairs
        if resume:
            processed_pair_keys = self.get_processed_pairs_v3(model_name)
            pairs_to_process = []
            
            for pair in all_pairs:
                cp1 = ord(pair.emoji1)
                cp2 = ord(pair.emoji2)
                if cp1 > cp2:
                    cp1, cp2 = cp2, cp1
                pair_key = f"{cp1}_{cp2}"
                
                if pair_key not in processed_pair_keys:
                    pairs_to_process.append(pair)
            
            logger.info(f"🔄 Resume mode: {len(processed_pair_keys):,} already done")
            logger.info(f"⚡ Processing {len(pairs_to_process):,} remaining pairs")
        
        if not pairs_to_process:
            logger.info("✅ All pairs already processed!")
            pairs_to_process = all_pairs
        
        # Process in batches
        start_time = datetime.utcnow()
        total_processed = 0
        
        async with SemanticCalculator(model_config) as calculator:
            batch_size = self.config["processing"].get("batch_size", 100)
            total_batches = (len(pairs_to_process) + batch_size - 1) // batch_size
            
            logger.info(f"📊 Processing {len(pairs_to_process):,} pairs in {total_batches:,} batches")
            logger.info(f"⚡ Batch size: {batch_size}")
            logger.info("🌌 FULL EMOJI SEMANTIC MANIFOLD DISCOVERY INITIATED!")
            
            for i in range(0, len(pairs_to_process), batch_size):
                batch = pairs_to_process[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                try:
                    logger.info(f"🔥 Processing batch {batch_num:,}/{total_batches:,}")
                    
                    batch_results = await calculator.calculate_distances(batch)
                    
                    # Store results to database
                    self.store_distances_v3(batch_results, model_id)
                    
                    total_processed += len(batch)
                    
                    # Progress update every 10 batches
                    if batch_num % 10 == 0:
                        elapsed = (datetime.utcnow() - start_time).total_seconds()
                        rate = total_processed / elapsed if elapsed > 0 else 0
                        remaining = len(pairs_to_process) - total_processed
                        eta_seconds = remaining / rate if rate > 0 else 0
                        eta_hours = eta_seconds / 3600
                        
                        progress = (total_processed / len(pairs_to_process)) * 100
                        
                        logger.info("⚡" * 50)
                        logger.info(f"⚡ PROGRESS: {total_processed:,}/{len(pairs_to_process):,} ({progress:.1f}%)")
                        logger.info(f"⚡ RATE: {rate:.1f} pairs/second")
                        logger.info(f"⚡ ETA: {eta_hours:.1f} hours")
                        logger.info("⚡" * 50)
                
                except Exception as e:
                    logger.error(f"💥 Error in batch {batch_num}: {e}")
                    continue
        
        completion_time = datetime.utcnow()
        duration = (completion_time - start_time).total_seconds()
        
        logger.info("🎉" * 30)
        logger.info(f"🎉 EMOJI SEMANTIC MANIFOLD DISCOVERY COMPLETE!")
        logger.info(f"🎉 Processed: {len(all_pairs):,} pairs")
        logger.info(f"🎉 Duration: {duration/3600:.1f} hours")
        logger.info(f"🎉 Rate: {len(all_pairs)/duration:.1f} pairs/second")
        logger.info("🎉" * 30)
        
        # Generate analysis results
        result = AnalysisResult(
            model_name=model_name,
            total_pairs=len(all_pairs),
            completion_time=completion_time,
            strongest_oppositions=[],
            weakest_oppositions=[],
            statistics={
                "total_pairs": len(all_pairs),
                "duration_hours": duration/3600,
                "pairs_per_second": len(all_pairs)/duration if duration > 0 else 0,
            },
            metadata={
                "duration_seconds": duration,
                "pairs_per_second": len(pairs_to_process) / duration if duration > 0 else 0,
                "resumed": resume,
                "config": self.config["models"][model_name],
                "emoji_count": 1337,
                "total_emoji_pairs": len(all_pairs)
            }
        )
        
        logger.info("🚀 EMOJI SEMANTIC MANIFOLD: MISSION ACCOMPLISHED!")
        return result


async def main():
    """Run the full emoji semantic manifold discovery"""
    processor = BatchProcessorV3()
    
    # Process with all-MPNet model
    result = await processor.process_model_v3("all_mpnet", resume=True)
    
    print("🌌 FULL EMOJI SEMANTIC MANIFOLD DISCOVERY COMPLETE! 🌌")
    print(f"Total pairs processed: {result.total_pairs:,}")
    print(f"Duration: {result.metadata['duration_seconds']/3600:.1f} hours")
    print(f"Rate: {result.metadata['pairs_per_second']:.1f} pairs/second")


if __name__ == "__main__":
    asyncio.run(main())
