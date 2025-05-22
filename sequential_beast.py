#!/usr/bin/env python3
"""
Sequential Beast Mode - Process ALL models in sequence
Let the M2 Ultra earn its keep!
"""

import asyncio
import time
from emojifold.batch_v3 import BatchProcessorV3

# The Beast Queue - M2 Ultra Challenge Mode
MODEL_QUEUE = [
    "all_mpnet",          # Currently running - 768 dims
    "e5_large",           # Microsoft's monster - 1024 dims  
    "bge_large",          # BAAI's beast - 1024 dims
    "instructor_xl",      # Instruction powerhouse - 768 dims
    "gte_large",          # Efficient large - 1024 dims
    "multilingual_e5",    # Global perspective - 1024 dims
]

async def run_sequential_beast():
    """
    Unleash the M2 Ultra on the complete model queue
    """
    processor = BatchProcessorV3()
    
    print("🚀" * 50)
    print("🚀 SEQUENTIAL BEAST MODE ACTIVATED")
    print("🚀 M2 Ultra: Time to earn your keep!")
    print("🚀" * 50)
    print()
    
    total_start = time.time()
    
    for i, model_name in enumerate(MODEL_QUEUE, 1):
        print(f"🔥 [{i}/{len(MODEL_QUEUE)}] Loading {model_name}...")
        
        # Visual loading indicator
        model_info = {
            "all_mpnet": "768 dims • 438MB",
            "e5_large": "1024 dims • 1.3GB", 
            "bge_large": "1024 dims • 1.3GB",
            "instructor_xl": "768 dims • 4.9GB",
            "gte_large": "1024 dims • 670MB",
            "multilingual_e5": "1024 dims • 2.2GB"
        }
        
        print(f"   📊 {model_info.get(model_name, 'Unknown specs')}")
        
        start_time = time.time()
        
        try:
            result = await processor.process_model_v3(model_name, resume=True)
            
            elapsed = time.time() - start_time
            
            print(f"✅ {model_name} COMPLETE!")
            print(f"   ⏱️  Duration: {elapsed/3600:.1f} hours")
            print(f"   📊 Pairs: {result.total_pairs:,}")
            print(f"   🚀 Rate: {result.metadata.get('pairs_per_second', 0):.1f} pairs/sec")
            print()
            
        except Exception as e:
            print(f"💥 ERROR with {model_name}: {e}")
            print("   🔄 Continuing with next model...")
            print()
            continue
    
    total_elapsed = time.time() - total_start
    
    print("🎉" * 50)
    print("🎉 BEAST MODE COMPLETE!")
    print(f"🎉 Total time: {total_elapsed/3600:.1f} hours")
    print(f"🎉 M2 Ultra status: FULLY UTILIZED")
    print("🎉" * 50)
    print()
    print("🌌 EMOJI SEMANTIC MULTIVERSE: UNLOCKED")
    print("📊 Cross-model analysis: READY")
    print("🎯 Centroid calculation: PENDING")

if __name__ == "__main__":
    asyncio.run(run_sequential_beast())
