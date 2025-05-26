#!/usr/bin/env python3
"""
Find the 'Kansas of Meaning' - the emoji closest to each model's centroid
"""

import sqlite3
import json
import numpy as np
from collections import defaultdict
import os

# Database path
DB_PATH = os.path.expanduser("~/.emojifold/emojifold.db")

def get_unique_emoji_embeddings(conn, model_id):
    """Extract unique emoji embeddings from the semantic_distances table"""
    
    # We need to collect all unique embeddings
    # Each emoji appears in multiple pairs, so we'll use a dict to store unique ones
    emoji_embeddings = {}
    
    # Get embeddings from emoji1 position
    query1 = """
    SELECT DISTINCT 
        e.codepoint,
        e.emoji,
        e.name,
        sd.embedding1
    FROM semantic_distances sd
    JOIN emoji_pairs ep ON sd.pair_id = ep.id
    JOIN emojis e ON ep.emoji1_codepoint = e.codepoint
    WHERE sd.model_id = ? 
        AND sd.embedding1 IS NOT NULL
    """
    
    cursor = conn.cursor()
    cursor.execute(query1, (model_id,))
    
    for row in cursor.fetchall():
        codepoint, emoji, name, embedding_json = row
        if embedding_json and codepoint not in emoji_embeddings:
            embedding = np.array(json.loads(embedding_json))
            emoji_embeddings[codepoint] = {
                'emoji': emoji,
                'name': name,
                'embedding': embedding
            }
    
    # Also get embeddings from emoji2 position
    query2 = """
    SELECT DISTINCT 
        e.codepoint,
        e.emoji,
        e.name,
        sd.embedding2
    FROM semantic_distances sd
    JOIN emoji_pairs ep ON sd.pair_id = ep.id
    JOIN emojis e ON ep.emoji2_codepoint = e.codepoint
    WHERE sd.model_id = ? 
        AND sd.embedding2 IS NOT NULL
    """
    
    cursor.execute(query2, (model_id,))
    
    for row in cursor.fetchall():
        codepoint, emoji, name, embedding_json = row
        if embedding_json and codepoint not in emoji_embeddings:
            embedding = np.array(json.loads(embedding_json))
            emoji_embeddings[codepoint] = {
                'emoji': emoji,
                'name': name,
                'embedding': embedding
            }
    
    return emoji_embeddings

def find_kansas_emoji(conn):
    """Find the emoji closest to centroid for each model"""
    
    # Get all models
    models_query = """
    SELECT m.id, m.name, m.embedding_dim
    FROM models m
    JOIN model_centroids mc ON m.id = mc.model_id
    WHERE mc.metric_type = 'euclidean'
    ORDER BY m.id
    """
    
    cursor = conn.cursor()
    cursor.execute(models_query)
    models = cursor.fetchall()
    
    results = []
    
    for model_id, model_name, embedding_dim in models:
        print(f"\nProcessing {model_name} (dim={embedding_dim})...")
        
        # Get centroid
        centroid_query = """
        SELECT centroid_vector
        FROM model_centroids
        WHERE model_id = ? AND metric_type = 'euclidean'
        """
        cursor.execute(centroid_query, (model_id,))
        centroid_json = cursor.fetchone()[0]
        centroid = np.array(json.loads(centroid_json))
        
        # Get all unique emoji embeddings
        emoji_embeddings = get_unique_emoji_embeddings(conn, model_id)
        print(f"  Found {len(emoji_embeddings)} unique emoji embeddings")
        
        # Calculate distances from centroid
        min_distance = float('inf')
        kansas_emoji = None
        
        distances = []
        
        for codepoint, data in emoji_embeddings.items():
            embedding = data['embedding']
            
            # Calculate Euclidean distance
            distance = np.linalg.norm(centroid - embedding)
            distances.append(distance)
            
            if distance < min_distance:
                min_distance = distance
                kansas_emoji = data
        
        # Also calculate some statistics
        distances = np.array(distances)
        
        results.append({
            'model': model_name,
            'dimension': embedding_dim,
            'kansas_emoji': kansas_emoji['emoji'],
            'kansas_name': kansas_emoji['name'],
            'distance_to_centroid': min_distance,
            'avg_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'total_emojis': len(emoji_embeddings)
        })
        
        print(f"  Kansas emoji: {kansas_emoji['emoji']} ({kansas_emoji['name']})")
        print(f"  Distance: {min_distance:.4f}")
    
    return results

def print_results(results):
    """Pretty print the results"""
    
    print("\n" + "="*80)
    print("🌾 THE KANSAS OF MEANING - Emojis Closest to Semantic Center 🌾")
    print("="*80)
    
    for r in results:
        print(f"\n{r['model']} ({r['dimension']}D):")
        print(f"  Kansas Emoji: {r['kansas_emoji']} - {r['kansas_name']}")
        print(f"  Distance from centroid: {r['distance_to_centroid']:.4f}")
        print(f"  Average emoji distance: {r['avg_distance']:.4f} ± {r['std_distance']:.4f}")
    
    print("\n" + "="*80)
    print("Interpretation:")
    print("- These emojis represent the 'semantic center' of each model")
    print("- Lower distance = closer to the average of all emoji meanings")
    print("- Different models may have different 'centers of meaning'")
    print("="*80)

def main():
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    
    try:
        # Find Kansas emojis
        results = find_kansas_emoji(conn)
        
        # Print results
        print_results(results)
        
        # Save results to a file
        with open('kansas_emojis.json', 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to kansas_emojis.json")
        
    finally:
        conn.close()

if __name__ == "__main__":
    main()