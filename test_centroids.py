#!/usr/bin/env python3
"""
Test script for centroid calculations
"""

import sqlite3
import json
import numpy as np

# Test with just one model first
db_path = "/Users/rob/repos/emojifold/data/emojifold.db"
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

# Get first model
cursor = conn.cursor()
cursor.execute("SELECT * FROM models LIMIT 1")
model = cursor.fetchone()
print(f"Testing with model: {model['name']}")

# Get a small sample of embeddings
cursor.execute("""
    SELECT embedding1, embedding2 
    FROM semantic_distances 
    WHERE model_id = ? AND embedding1 IS NOT NULL AND embedding2 IS NOT NULL
    LIMIT 100
""", (model['id'],))

embeddings = []
for row in cursor.fetchall():
    emb1 = np.array(json.loads(row['embedding1']))
    emb2 = np.array(json.loads(row['embedding2']))
    embeddings.extend([emb1, emb2])

print(f"Got {len(embeddings)} embeddings")
print(f"Embedding dimension: {len(embeddings[0])}")

# Calculate centroid
embedding_matrix = np.array(embeddings)
centroid = np.mean(embedding_matrix, axis=0)

print(f"Centroid calculated: {centroid[:5]}... (showing first 5 dimensions)")

# Calculate some basic stats
distances_from_centroid = [np.linalg.norm(emb - centroid) for emb in embeddings]
mean_distance = np.mean(distances_from_centroid)

print(f"Mean distance from centroid: {mean_distance:.6f}")
print("✓ Basic calculation works!")

conn.close()
