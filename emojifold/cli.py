"""
Command-line interface for EmojiFold
"""

import asyncio
import click
import logging
import sys
from pathlib import Path

from .batch import BatchProcessor
from .calculator import quick_distance
from .storage import EmojiDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('./logs/emojifold.log')
    ]
)

logger = logging.getLogger(__name__)


@click.group()
def main():
    """EmojiFold: Emoji Semantic Manifold Discovery"""
    pass


@main.command()
@click.option('--emoji1', required=True, help='First emoji')
@click.option('--emoji2', required=True, help='Second emoji')
@click.option('--model', default='all_minilm', help='Model to use')
def distance(emoji1: str, emoji2: str, model: str):
    """Calculate semantic distance between two emoji"""
    async def _calculate():
        dist = await quick_distance(emoji1, emoji2, model)
        click.echo(f"Distance between {emoji1} and {emoji2}: {dist:.4f}")
        click.echo(f"Similarity: {1-dist:.4f}")
    
    asyncio.run(_calculate())


@main.command()
@click.option('--model', help='Specific model to process (default: all models)')
@click.option('--test', is_flag=True, help='Run in test mode with limited emoji set')
@click.option('--no-resume', is_flag=True, help='Don\'t resume from previous run')
def batch(model: str, test: bool, no_resume: bool):
    """Run batch processing for emoji semantic analysis"""
    async def _batch_process():
        processor = BatchProcessor()
        
        if model:
            # Process single model
            result = await processor.process_model(
                model, 
                test_mode=test, 
                resume=not no_resume
            )
            click.echo(f"\\nCompleted {model}:")
            click.echo(f"  Processed: {result.total_pairs} pairs")
            click.echo(f"  Duration: {result.metadata['duration_seconds']:.1f}s")
            click.echo(f"  Rate: {result.metadata['pairs_per_second']:.1f} pairs/sec")
            click.echo(f"  Max distance: {result.statistics['max_distance']:.4f}")
            click.echo(f"  Mean distance: {result.statistics['mean_distance']:.4f}")
        else:
            # Process all models
            results = await processor.process_all_models(test_mode=test)
            
            click.echo(f"\\nCompleted batch processing for {len(results)} models:")
            for model_name, result in results.items():
                click.echo(f"\\n{model_name}:")
                click.echo(f"  Processed: {result.total_pairs} pairs")
                click.echo(f"  Duration: {result.metadata['duration_seconds']:.1f}s")
                click.echo(f"  Rate: {result.metadata['pairs_per_second']:.1f} pairs/sec")
                click.echo(f"  Max distance: {result.statistics['max_distance']:.4f}")
    
    asyncio.run(_batch_process())


@main.command()
@click.option('--model', required=True, help='Model to analyze')
@click.option('--limit', default=20, help='Number of results to show')
def top(model: str, limit: int):
    """Show top semantic oppositions for a model"""
    db = EmojiDatabase()
    
    strongest = db.get_strongest_oppositions(model, limit=limit)
    stats = db.get_model_statistics(model)
    
    click.echo(f"\\nTop {limit} semantic oppositions for {model}:")
    click.echo(f"Total pairs analyzed: {stats['total_pairs']}")
    click.echo(f"Mean distance: {stats['mean_distance']:.4f}")
    click.echo(f"Max distance: {stats['max_distance']:.4f}")
    click.echo("\\n" + "="*60)
    
    for i, result in enumerate(strongest, 1):
        click.echo(f"{i:2d}. {result['emoji1']} vs {result['emoji2']} - {result['distance']:.4f}")
        if result['description1'] and result['description2']:
            click.echo(f"    ({result['description1']} vs {result['description2']})")


@main.command()
@click.option('--model', required=True, help='Model to export')
@click.option('--output', help='Output file path')
def export(model: str, output: str):
    """Export results to JSON file"""
    db = EmojiDatabase()
    
    if not output:
        output = f"./results/{model}_export.json"
    
    db.export_results(model, output)
    click.echo(f"Results exported to {output}")


@main.command()
@click.option('--emoji1', required=True, help='First emoji')
@click.option('--emoji2', required=True, help='Second emoji')
def compare(emoji1: str, emoji2: str):
    """Compare emoji pair across all models"""
    from .models import EmojiPair
    
    db = EmojiDatabase()
    pair = EmojiPair(emoji1, emoji2)
    
    results = db.get_cross_model_comparison(pair.pair_id)
    
    if not results:
        click.echo(f"No results found for {emoji1} vs {emoji2}")
        return
    
    click.echo(f"\\nCross-model comparison for {emoji1} vs {emoji2}:")
    click.echo("="*50)
    
    for result in results:
        click.echo(f"{result['model_name']:20s} | {result['distance']:.4f} | {result['similarity']:.4f}")


@main.command()
def setup():
    """Set up project directories and configuration"""
    import os
    
    # Create directories from config
    config_dirs = [
        "~/.emojifold",
        "~/.emojifold/results", 
        "~/.emojifold/cache",
        "./logs"  # Keep logs in project for development
    ]
    
    for dir_path in config_dirs:
        expanded_path = Path(dir_path).expanduser()
        expanded_path.mkdir(parents=True, exist_ok=True)
        click.echo(f"Created directory: {expanded_path}")
    
    # Set up HuggingFace cache
    hf_cache = Path("~/.cache/huggingface").expanduser()
    hf_cache.mkdir(parents=True, exist_ok=True)
    click.echo(f"HuggingFace cache: {hf_cache}")
    
    # Copy .env.example to .env if it doesn't exist
    env_path = Path(".env")
    if not env_path.exists():
        env_example = Path(".env.example")
        if env_example.exists():
            env_path.write_text(env_example.read_text())
            click.echo("Created .env from .env.example")
        else:
            click.echo("Warning: .env.example not found")
    
    click.echo("\\nSetup complete! Next steps:")
    click.echo("1. Edit .env file with your configuration")
    click.echo("2. Edit config.yaml to customize models and processing")
    click.echo("3. Ensure Ollama is running if using Ollama models")
    click.echo("4. Run: emojifold batch --test  # for test run")
    click.echo("5. Run: emojifold batch        # for full analysis")


if __name__ == '__main__':
    main()
