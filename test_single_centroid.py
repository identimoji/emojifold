#!/usr/bin/env python3
"""
Quick test to calculate one model centroid
"""
import sys
sys.path.append('/Users/rob/repos/emojifold')

import sqlite3
import json
import numpy as np
from datetime import datetime

def test_single_model_centroid():
    db_path = "/Users/rob/.emojifold/emojifold.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    print("🔍 Testing single model centroid calculation...")
    
    # Get first model
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM models ORDER BY id LIMIT 1")
    model = cursor.fetchone()
    print(f"📊 Model: {model['name']} (dim: {model['embedding_dim']})")
    
    # Get sample of embeddings (limit to 1000 for quick test)
    print("📥 Loading embeddings...")
    cursor.execute("""
        SELECT embedding1, embedding2 
        FROM semantic_distances 
        WHERE model_id = ? AND embedding1 IS NOT NULL AND embedding2 IS NOT NULL
        LIMIT 500
    """, (model['id'],))
    
    embeddings = []
    for row in cursor.fetchall():
        emb1 = np.array(json.loads(row['embedding1']))
        emb2 = np.array(json.loads(row['embedding2']))
        embeddings.extend([emb1, emb2])
    
    print(f"✅ Loaded {len(embeddings)} embeddings")
    
    # Calculate centroid
    print("🧮 Calculating centroid...")
    embedding_matrix = np.array(embeddings)
    centroid = np.mean(embedding_matrix, axis=0)
    
    # Calculate stats
    distances = [np.linalg.norm(emb - centroid) for emb in embeddings]
    
    stats = {
        'centroid_sample': centroid[:5].tolist(),  # First 5 dimensions
        'mean_distance': float(np.mean(distances)),
        'std_distance': float(np.std(distances)),
        'embedding_count': len(embeddings),
        'dimension': len(centroid)
    }
    
    print("📈 Results:")
    print(f"  Centroid (first 5 dims): {[f'{x:.4f}' for x in stats['centroid_sample']]}")
    print(f"  Mean distance from centroid: {stats['mean_distance']:.6f}")
    print(f"  Std distance: {stats['std_distance']:.6f}")
    print(f"  Total embeddings: {stats['embedding_count']}")
    print(f"  Embedding dimensions: {stats['dimension']}")
    
    # Test storing in database
    print("\n💾 Testing database storage...")
    centroid_json = json.dumps(centroid.tolist())
    extrema_json = json.dumps(stats)
    
    cursor.execute("""
        INSERT OR REPLACE INTO model_centroids 
        (model_id, metric_type, centroid_vector, extrema_stats, version, calculated_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        model['id'],
        'euclidean_test',
        centroid_json,
        extrema_json,
        1,
        datetime.now().isoformat()
    ))
    
    conn.commit()
    print("✅ Centroid stored successfully!")
    
    # Verify storage
    cursor.execute("""
        SELECT * FROM model_centroids 
        WHERE model_id = ? AND metric_type = ?
    """, (model['id'], 'euclidean_test'))
    
    stored = cursor.fetchone()
    if stored:
        print("✅ Storage verified!")
        stored_centroid = json.loads(stored['centroid_vector'])
        print(f"  Stored centroid (first 3): {[f'{x:.4f}' for x in stored_centroid[:3]]}")
    
    conn.close()
    print("\n🎉 Test completed successfully!")

if __name__ == '__main__':
    test_single_model_centroid()
