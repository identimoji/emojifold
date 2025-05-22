"""
Batch processing for large-scale emoji semantic analysis
"""

import asyncio
import logging
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import unicodedata

from .models import EmojiPair, SemanticDistance, AnalysisResult, ModelConfig
from .calculator import SemanticCalculator
from .storage import EmojiDatabase

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Process large batches of emoji pairs for semantic analysis"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        
        # Expand home directory paths
        db_path = Path(self.config["storage"]["database_path"]).expanduser()
        self.db = EmojiDatabase(str(db_path))
        
        # Ensure output directories exist
        results_dir = Path(self.config["storage"]["results_dir"]).expanduser()
        cache_dir = Path(self.config["storage"]["cache_dir"]).expanduser()
        
        results_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        Path("./logs").mkdir(parents=True, exist_ok=True)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def generate_emoji_pairs(self, test_mode: bool = False) -> List[EmojiPair]:
        """Generate all possible emoji pairs for analysis"""
        emojis = self._get_emoji_list()
        
        if test_mode:
            # Use a smaller test set
            test_size = self.config["emojis"]["test_set_size"]
            emojis = emojis[:test_size]
            logger.info(f"Test mode: using {len(emojis)} emojis")
        
        pairs = []
        total_pairs = len(emojis) * (len(emojis) - 1) // 2
        logger.info(f"Generating {total_pairs} emoji pairs from {len(emojis)} emojis")
        
        for i, emoji1 in enumerate(emojis):
            for emoji2 in emojis[i+1:]:
                pair = EmojiPair(
                    emoji1=emoji1["emoji"],
                    emoji2=emoji2["emoji"],
                    description1=emoji1.get("description"),
                    description2=emoji2.get("description")
                )
                pairs.append(pair)
        
        logger.info(f"Generated {len(pairs)} emoji pairs")
        return pairs
    
    def _get_emoji_list(self) -> List[Dict[str, str]]:
        """Get list of emoji with descriptions"""
        # This is a simplified version - in practice you'd want to load from
        # a comprehensive emoji database like Unicode CLDR
        
        # For now, let's use a curated list of common emojis
        common_emojis = [
            # Faces & Emotions
            {"emoji": "😀", "description": "grinning face"},
            {"emoji": "😃", "description": "grinning face with big eyes"},
            {"emoji": "😄", "description": "grinning face with smiling eyes"},
            {"emoji": "😁", "description": "beaming face with smiling eyes"},
            {"emoji": "😆", "description": "grinning squinting face"},
            {"emoji": "😅", "description": "grinning face with sweat"},
            {"emoji": "🤣", "description": "rolling on the floor laughing"},
            {"emoji": "😂", "description": "face with tears of joy"},
            {"emoji": "🙂", "description": "slightly smiling face"},
            {"emoji": "🙃", "description": "upside down face"},
            {"emoji": "😉", "description": "winking face"},
            {"emoji": "😊", "description": "smiling face with smiling eyes"},
            {"emoji": "😇", "description": "smiling face with halo"},
            {"emoji": "😍", "description": "smiling face with heart eyes"},
            {"emoji": "🤩", "description": "star struck"},
            {"emoji": "😘", "description": "face blowing a kiss"},
            {"emoji": "😗", "description": "kissing face"},
            {"emoji": "☺️", "description": "smiling face"},
            {"emoji": "😚", "description": "kissing face with closed eyes"},
            {"emoji": "😙", "description": "kissing face with smiling eyes"},
            {"emoji": "😋", "description": "face savoring food"},
            {"emoji": "😛", "description": "face with tongue"},
            {"emoji": "😜", "description": "winking face with tongue"},
            {"emoji": "🤪", "description": "zany face"},
            {"emoji": "😝", "description": "squinting face with tongue"},
            {"emoji": "🤑", "description": "money mouth face"},
            {"emoji": "🤗", "description": "hugging face"},
            {"emoji": "🤭", "description": "face with hand over mouth"},
            {"emoji": "🤫", "description": "shushing face"},
            {"emoji": "🤔", "description": "thinking face"},
            {"emoji": "🤐", "description": "zipper mouth face"},
            {"emoji": "🤨", "description": "face with raised eyebrow"},
            {"emoji": "😐", "description": "neutral face"},
            {"emoji": "😑", "description": "expressionless face"},
            {"emoji": "😶", "description": "face without mouth"},
            {"emoji": "😏", "description": "smirking face"},
            {"emoji": "😒", "description": "unamused face"},
            {"emoji": "🙄", "description": "face with rolling eyes"},
            {"emoji": "😬", "description": "grimacing face"},
            {"emoji": "🤥", "description": "lying face"},
            {"emoji": "😔", "description": "pensive face"},
            {"emoji": "😪", "description": "sleepy face"},
            {"emoji": "🤤", "description": "drooling face"},
            {"emoji": "😴", "description": "sleeping face"},
            {"emoji": "😷", "description": "face with medical mask"},
            {"emoji": "🤒", "description": "face with thermometer"},
            {"emoji": "🤕", "description": "face with head bandage"},
            {"emoji": "🤢", "description": "nauseated face"},
            {"emoji": "🤮", "description": "face vomiting"},
            {"emoji": "🤧", "description": "sneezing face"},
            {"emoji": "🥵", "description": "hot face"},
            {"emoji": "🥶", "description": "cold face"},
            {"emoji": "🥴", "description": "woozy face"},
            {"emoji": "😵", "description": "dizzy face"},
            {"emoji": "🤯", "description": "exploding head"},
            {"emoji": "🤠", "description": "cowboy hat face"},
            {"emoji": "🥳", "description": "partying face"},
            {"emoji": "🥸", "description": "disguised face"},
            {"emoji": "😎", "description": "smiling face with sunglasses"},
            {"emoji": "🤓", "description": "nerd face"},
            {"emoji": "🧐", "description": "face with monocle"},
            {"emoji": "😕", "description": "confused face"},
            {"emoji": "😟", "description": "worried face"},
            {"emoji": "🙁", "description": "slightly frowning face"},
            {"emoji": "☹️", "description": "frowning face"},
            {"emoji": "😮", "description": "face with open mouth"},
            {"emoji": "😯", "description": "hushed face"},
            {"emoji": "😲", "description": "astonished face"},
            {"emoji": "😳", "description": "flushed face"},
            {"emoji": "🥺", "description": "pleading face"},
            {"emoji": "😦", "description": "frowning face with open mouth"},
            {"emoji": "😧", "description": "anguished face"},
            {"emoji": "😨", "description": "fearful face"},
            {"emoji": "😰", "description": "anxious face with sweat"},
            {"emoji": "😥", "description": "sad but relieved face"},
            {"emoji": "😢", "description": "crying face"},
            {"emoji": "😭", "description": "loudly crying face"},
            {"emoji": "😱", "description": "face screaming in fear"},
            {"emoji": "😖", "description": "confounded face"},
            {"emoji": "😣", "description": "persevering face"},
            {"emoji": "😞", "description": "disappointed face"},
            {"emoji": "😓", "description": "downcast face with sweat"},
            {"emoji": "😩", "description": "weary face"},
            {"emoji": "😫", "description": "tired face"},
            {"emoji": "🥱", "description": "yawning face"},
            {"emoji": "😤", "description": "face with steam from nose"},
            {"emoji": "😡", "description": "pouting face"},
            {"emoji": "😠", "description": "angry face"},
            {"emoji": "🤬", "description": "face with symbols on mouth"},
            {"emoji": "😈", "description": "smiling face with horns"},
            {"emoji": "👿", "description": "angry face with horns"},
            {"emoji": "💀", "description": "skull"},
            {"emoji": "☠️", "description": "skull and crossbones"},
            
            # Colors
            {"emoji": "🔴", "description": "red circle"},
            {"emoji": "🟠", "description": "orange circle"},
            {"emoji": "🟡", "description": "yellow circle"},
            {"emoji": "🟢", "description": "green circle"},
            {"emoji": "🔵", "description": "blue circle"},
            {"emoji": "🟣", "description": "purple circle"},
            {"emoji": "🟤", "description": "brown circle"},
            {"emoji": "⚫", "description": "black circle"},
            {"emoji": "⚪", "description": "white circle"},
            
            # Symbols
            {"emoji": "➕", "description": "plus"},
            {"emoji": "➖", "description": "minus"},
            {"emoji": "✖️", "description": "multiply"},
            {"emoji": "➗", "description": "divide"},
            {"emoji": "⬆️", "description": "up arrow"},
            {"emoji": "⬇️", "description": "down arrow"},
            {"emoji": "⬅️", "description": "left arrow"},
            {"emoji": "➡️", "description": "right arrow"},
            
            # Nature & Weather
            {"emoji": "🌞", "description": "sun with face"},
            {"emoji": "🌙", "description": "crescent moon"},
            {"emoji": "⭐", "description": "star"},
            {"emoji": "🌟", "description": "glowing star"},
            {"emoji": "🔥", "description": "fire"},
            {"emoji": "❄️", "description": "snowflake"},
            {"emoji": "🌊", "description": "water wave"},
            {"emoji": "⚡", "description": "lightning bolt"},
            
            # Animals
            {"emoji": "🐶", "description": "dog face"},
            {"emoji": "🐱", "description": "cat face"},
            {"emoji": "🐭", "description": "mouse face"},
            {"emoji": "🐹", "description": "hamster face"},
            {"emoji": "🐰", "description": "rabbit face"},
            {"emoji": "🦊", "description": "fox face"},
            {"emoji": "🐻", "description": "bear face"},
            {"emoji": "🐼", "description": "panda face"},
            {"emoji": "🐨", "description": "koala"},
            {"emoji": "🐯", "description": "tiger face"},
            {"emoji": "🦁", "description": "lion face"},
            {"emoji": "🐮", "description": "cow face"},
            {"emoji": "🐷", "description": "pig face"},
            {"emoji": "🐸", "description": "frog face"},
            {"emoji": "🐵", "description": "monkey face"},
            {"emoji": "🐔", "description": "chicken"},
            {"emoji": "🐧", "description": "penguin"},
            {"emoji": "🐦", "description": "bird"},
            {"emoji": "🐤", "description": "baby chick"},
            {"emoji": "🐣", "description": "hatching chick"},
            {"emoji": "🐥", "description": "front facing baby chick"},
            {"emoji": "🦆", "description": "duck"},
            {"emoji": "🦅", "description": "eagle"},
            {"emoji": "🦉", "description": "owl"},
            {"emoji": "🦇", "description": "bat"},
            {"emoji": "🐺", "description": "wolf face"},
            {"emoji": "🐗", "description": "boar"},
            {"emoji": "🐴", "description": "horse face"},
            {"emoji": "🦄", "description": "unicorn face"},
            {"emoji": "🐝", "description": "honeybee"},
            {"emoji": "🐛", "description": "bug"},
            {"emoji": "🦋", "description": "butterfly"},
            {"emoji": "🐌", "description": "snail"},
            {"emoji": "🐞", "description": "lady beetle"},
            {"emoji": "🐜", "description": "ant"},
            {"emoji": "🦟", "description": "mosquito"},
            {"emoji": "🕷️", "description": "spider"},
            {"emoji": "🕸️", "description": "spider web"},
            {"emoji": "🦂", "description": "scorpion"},
            {"emoji": "🐢", "description": "turtle"},
            {"emoji": "🐍", "description": "snake"},
            {"emoji": "🦎", "description": "lizard"},
            {"emoji": "🦖", "description": "t-rex"},
            {"emoji": "🦕", "description": "sauropod"},
            {"emoji": "🐙", "description": "octopus"},
            {"emoji": "🦑", "description": "squid"},
            {"emoji": "🦐", "description": "shrimp"},
            {"emoji": "🦞", "description": "lobster"},
            {"emoji": "🦀", "description": "crab"},
            {"emoji": "🐡", "description": "blowfish"},
            {"emoji": "🐠", "description": "tropical fish"},
            {"emoji": "🐟", "description": "fish"},
            {"emoji": "🐬", "description": "dolphin"},
            {"emoji": "🐳", "description": "spouting whale"},
            {"emoji": "🐋", "description": "whale"},
            {"emoji": "🦈", "description": "shark"},
            {"emoji": "🐊", "description": "crocodile"},
            {"emoji": "🐅", "description": "tiger"},
            {"emoji": "🐆", "description": "leopard"},
            {"emoji": "🦓", "description": "zebra"},
            {"emoji": "🦍", "description": "gorilla"},
            {"emoji": "🦧", "description": "orangutan"},
            {"emoji": "🐘", "description": "elephant"},
            {"emoji": "🦛", "description": "hippopotamus"},
            {"emoji": "🦏", "description": "rhinoceros"},
            {"emoji": "🐪", "description": "camel"},
            {"emoji": "🐫", "description": "two hump camel"},
            {"emoji": "🦒", "description": "giraffe"},
            {"emoji": "🦘", "description": "kangaroo"},
            {"emoji": "🐃", "description": "water buffalo"},
            {"emoji": "🐂", "description": "ox"},
            {"emoji": "🐄", "description": "cow"},
            {"emoji": "🐎", "description": "horse"},
            {"emoji": "🐖", "description": "pig"},
            {"emoji": "🐏", "description": "ram"},
            {"emoji": "🐑", "description": "ewe"},
            {"emoji": "🦙", "description": "llama"},
            {"emoji": "🐐", "description": "goat"},
            {"emoji": "🦌", "description": "deer"},
            {"emoji": "🐕", "description": "dog"},
            {"emoji": "🐩", "description": "poodle"},
            {"emoji": "🦮", "description": "guide dog"},
            {"emoji": "🐕‍🦺", "description": "service dog"},
            {"emoji": "🐈", "description": "cat"},
            {"emoji": "🐈‍⬛", "description": "black cat"},
            {"emoji": "🐓", "description": "rooster"},
            {"emoji": "🦃", "description": "turkey"},
            {"emoji": "🦚", "description": "peacock"},
            {"emoji": "🦜", "description": "parrot"},
            {"emoji": "🦢", "description": "swan"},
            {"emoji": "🦩", "description": "flamingo"},
            {"emoji": "🕊️", "description": "dove"},
            {"emoji": "🐇", "description": "rabbit"},
            {"emoji": "🦝", "description": "raccoon"},
            {"emoji": "🦨", "description": "skunk"},
            {"emoji": "🦡", "description": "badger"},
            {"emoji": "🦫", "description": "beaver"},
            {"emoji": "🦦", "description": "otter"},
            {"emoji": "🦥", "description": "sloth"},
            {"emoji": "🐁", "description": "mouse"},
            {"emoji": "🐀", "description": "rat"},
            {"emoji": "🐿️", "description": "chipmunk"},
            {"emoji": "🦔", "description": "hedgehog"},
        ]
        
        return common_emojis
    
    async def process_model(
        self, 
        model_name: str, 
        test_mode: bool = False,
        resume: bool = True
    ) -> AnalysisResult:
        """Process all emoji pairs for a specific model"""
        logger.info(f"Starting batch processing for model: {model_name}")
        
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
        
        # Generate emoji pairs
        all_pairs = self.generate_emoji_pairs(test_mode=test_mode)
        
        # Filter out already processed pairs if resuming
        pairs_to_process = all_pairs
        if resume:
            processed_pairs = self.db.get_processed_pairs(model_name)
            pairs_to_process = [p for p in all_pairs if p.pair_id not in processed_pairs]
            logger.info(f"Resuming: {len(processed_pairs)} already done, {len(pairs_to_process)} remaining")
        
        if not pairs_to_process:
            logger.info("All pairs already processed!")
            # Still generate results for already processed data
            pairs_to_process = all_pairs
        
        # Process in batches
        start_time = datetime.utcnow()
        all_results = []
        
        async with SemanticCalculator(model_config) as calculator:
            batch_size = self.config["processing"].get("batch_size", 100)
            
            for i in range(0, len(pairs_to_process), batch_size):
                batch = pairs_to_process[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(pairs_to_process) + batch_size - 1)//batch_size}")
                
                try:
                    batch_results = await calculator.calculate_distances(batch)
                    all_results.extend(batch_results)
                    
                    # Store results to database
                    self.db.store_distances(batch_results)
                    
                    # Progress update
                    if i % (batch_size * 10) == 0:
                        logger.info(f"Completed {i + len(batch)}/{len(pairs_to_process)} pairs")
                
                except Exception as e:
                    logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                    continue
        
        completion_time = datetime.utcnow()
        duration = (completion_time - start_time).total_seconds()
        
        # Generate analysis results
        statistics = self.db.get_model_statistics(model_name)
        strongest = self.db.get_strongest_oppositions(model_name, limit=100)
        weakest = self.db.get_strongest_oppositions(model_name, limit=100)  # TODO: get weakest
        
        result = AnalysisResult(
            model_name=model_name,
            total_pairs=len(all_pairs),
            completion_time=completion_time,
            strongest_oppositions=[],  # Simplified for now
            weakest_oppositions=[],
            statistics=statistics,
            metadata={
                "duration_seconds": duration,
                "pairs_per_second": len(pairs_to_process) / duration if duration > 0 else 0,
                "test_mode": test_mode,
                "resumed": resume,
                "config": self.config["models"][model_name]
            }
        )
        
        # Store analysis result
        self.db.store_analysis_result(result)
        
        # Export results
        results_dir = Path(self.config["storage"]["results_dir"]).expanduser()
        output_path = results_dir / f"{model_name}_results.json"
        self.db.export_results(model_name, str(output_path))
        
        logger.info(f"Completed analysis for {model_name}: {len(all_pairs)} pairs in {duration:.1f}s")
        return result
    
    async def process_all_models(self, test_mode: bool = False) -> Dict[str, AnalysisResult]:
        """Process all configured models"""
        results = {}
        
        for model_name in self.config["models"]:
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"Processing model: {model_name}")
                logger.info(f"{'='*50}")
                
                result = await self.process_model(model_name, test_mode=test_mode)
                results[model_name] = result
                
            except Exception as e:
                logger.error(f"Failed to process model {model_name}: {e}")
                continue
        
        return results
