"""
CLI V3 - Full Emoji Semantic Manifold Discovery
"""

import asyncio
import click
import logging
import sys
from pathlib import Path

from emojifold.batch_v3 import BatchProcessorV3

# Configure logging for MAXIMUM POWER
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('./logs/emojifold-v3.log')
    ]
)

logger = logging.getLogger(__name__)


@click.group()
def main():
    """🚀 EmojiFold V3: Full Emoji Semantic Manifold Discovery"""
    pass


@main.command()
@click.option('--model', default='all_minilm', help='Model to use for processing')
@click.option('--no-resume', is_flag=True, help='Start fresh (don\'t resume)')
def manifold(model: str, no_resume: bool):
    """🚀 Launch FULL emoji semantic manifold discovery (893,116 pairs!)"""
    
    click.echo("🚀" * 30)
    click.echo("🚀 EMOJI SEMANTIC MANIFOLD DISCOVERY V3")
    click.echo("🚀 MAXIMUM POWER MODE ACTIVATED!")  
    click.echo(f"🚀 Target: 893,116 emoji pairs")
    click.echo(f"🚀 Model: {model}")
    click.echo(f"🚀 Resume: {'No' if no_resume else 'Yes'}")
    click.echo("🚀" * 30)
    
    if not no_resume:
        click.echo("⚡ Resume mode: Will continue from last processed pair")
    else:
        click.echo("🔥 Fresh start: Processing all pairs from beginning")
    
    click.echo("⏰ Estimated duration: 24-48 hours")
    click.echo("💾 Database: ~/.emojifold/emojifold_v3.db")
    
    # Confirm before launching
    if not click.confirm("🚀 Launch the FULL emoji semantic manifold discovery?"):
        click.echo("❌ Mission aborted")
        return
    
    async def _launch_manifold():
        processor = BatchProcessorV3()
        
        try:
            result = await processor.process_model_v3(model, resume=not no_resume)
            
            click.echo("🎉" * 40)
            click.echo("🎉 EMOJI SEMANTIC MANIFOLD DISCOVERY COMPLETE!")
            click.echo(f"🎉 Processed: {result.total_pairs:,} pairs")
            click.echo(f"🎉 Duration: {result.metadata['duration_seconds']/3600:.1f} hours")
            click.echo(f"🎉 Rate: {result.metadata['pairs_per_second']:.1f} pairs/sec")
            click.echo("🎉" * 40)
            
        except Exception as e:
            logger.error(f"💥 Mission failed: {e}")
            click.echo(f"💥 Error: {e}")
    
    asyncio.run(_launch_manifold())


@main.command()
def status():
    """📊 Check progress of emoji manifold discovery"""
    import sqlite3
    
    db_path = Path("~/.emojifold/emojifold_v3.db").expanduser()
    
    if not db_path.exists():
        click.echo("❌ No V3 database found")
        return
    
    with sqlite3.connect(db_path) as conn:
        # Count total emoji
        cursor = conn.execute("SELECT COUNT(*) FROM emojis")
        total_emoji = cursor.fetchone()[0]
        
        # Count processed pairs
        cursor = conn.execute("SELECT COUNT(*) FROM semantic_distances")
        processed_pairs = cursor.fetchone()[0]
        
        # Calculate theoretical max
        max_pairs = total_emoji * (total_emoji - 1) // 2
        
        # Get latest processing time
        cursor = conn.execute("""
            SELECT MAX(calculated_at) FROM semantic_distances
        """)
        latest = cursor.fetchone()[0]
        
        progress = (processed_pairs / max_pairs) * 100 if max_pairs > 0 else 0
        
        click.echo("📊" * 30)
        click.echo("📊 EMOJI MANIFOLD DISCOVERY STATUS")
        click.echo("📊" * 30)
        click.echo(f"📊 Total emoji: {total_emoji:,}")
        click.echo(f"📊 Max possible pairs: {max_pairs:,}")
        click.echo(f"📊 Processed pairs: {processed_pairs:,}")
        click.echo(f"📊 Progress: {progress:.2f}%")
        click.echo(f"📊 Latest processing: {latest}")
        
        if processed_pairs > 0:
            click.echo(f"📊 Remaining: {max_pairs - processed_pairs:,} pairs")


@main.command()
@click.option('--limit', default=20, help='Number of results to show')
def top(limit: int):
    """🏆 Show strongest emoji semantic oppositions discovered"""
    import sqlite3
    
    db_path = Path("~/.emojifold/emojifold_v3.db").expanduser()
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("""
            SELECT 
                e1.emoji, e1.name, e1.category,
                e2.emoji, e2.name, e2.category,
                sd.distance, sd.similarity
            FROM semantic_distances sd
            JOIN emoji_pairs ep ON sd.pair_id = ep.id
            JOIN emojis e1 ON ep.emoji1_codepoint = e1.codepoint
            JOIN emojis e2 ON ep.emoji2_codepoint = e2.codepoint
            ORDER BY sd.distance DESC
            LIMIT ?
        """, (limit,))
        
        results = cursor.fetchall()
        
        if not results:
            click.echo("❌ No results found")
            return
        
        click.echo("🏆" * 50)
        click.echo("🏆 STRONGEST EMOJI SEMANTIC OPPOSITIONS")
        click.echo("🏆" * 50)
        
        for i, (e1, n1, c1, e2, n2, c2, dist, sim) in enumerate(results, 1):
            click.echo(f"{i:2d}. {e1} vs {e2} - Distance: {dist:.4f}")
            click.echo(f"    {n1} ({c1}) ↔ {n2} ({c2})")
            click.echo("")


if __name__ == '__main__':
    main()
