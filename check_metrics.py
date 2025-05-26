#!/usr/bin/env python3
"""
Check what distance metrics we have available
"""
import sqlite3

db_path = "/Users/rob/.emojifold/emojifold.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("🔍 Checking available distance metrics...")

# Check which columns have data
cursor.execute("""
    SELECT 
        COUNT(CASE WHEN distance IS NOT NULL THEN 1 END) as euclidean_count,
        COUNT(CASE WHEN similarity IS NOT NULL THEN 1 END) as cosine_count,
        COUNT(CASE WHEN manhattan_distance IS NOT NULL THEN 1 END) as manhattan_count,
        COUNT(CASE WHEN dot_product IS NOT NULL THEN 1 END) as dot_count,
        COUNT(*) as total_rows
    FROM semantic_distances
""")

counts = cursor.fetchone()
print(f"\n📊 Available metrics across all rows:")
print(f"  Euclidean distance: {counts[0]:,} / {counts[4]:,}")
print(f"  Cosine similarity:  {counts[1]:,} / {counts[4]:,}")
print(f"  Manhattan distance: {counts[2]:,} / {counts[4]:,}")
print(f"  Dot product:        {counts[3]:,} / {counts[4]:,}")

# Check per model
print(f"\n🤖 Per-model breakdown:")
cursor.execute("""
    SELECT 
        m.id,
        m.name,
        COUNT(CASE WHEN sd.distance IS NOT NULL THEN 1 END) as euclidean,
        COUNT(CASE WHEN sd.similarity IS NOT NULL THEN 1 END) as cosine,
        COUNT(CASE WHEN sd.manhattan_distance IS NOT NULL THEN 1 END) as manhattan,
        COUNT(CASE WHEN sd.dot_product IS NOT NULL THEN 1 END) as dot_product
    FROM models m
    LEFT JOIN semantic_distances sd ON m.id = sd.model_id
    WHERE m.id IN (1, 2, 3, 4, 5)  -- Only complete models
    GROUP BY m.id, m.name
    ORDER BY m.id
""")

for row in cursor.fetchall():
    model_id, name, euclidean, cosine, manhattan, dot = row
    print(f"  {model_id}: {name}")
    print(f"    Euclidean: {euclidean:,}, Cosine: {cosine:,}")
    print(f"    Manhattan: {manhattan:,}, Dot: {dot:,}")

# Sample a few rows to see what data looks like
print(f"\n🔬 Sample data structure:")
cursor.execute("""
    SELECT distance, similarity, manhattan_distance, dot_product 
    FROM semantic_distances 
    WHERE model_id = 1 
    LIMIT 3
""")

for i, row in enumerate(cursor.fetchall(), 1):
    distance, similarity, manhattan, dot = row
    print(f"  Row {i}: dist={distance:.4f if distance else 'NULL'}, sim={similarity:.4f if similarity else 'NULL'}, man={manhattan or 'NULL'}, dot={dot or 'NULL'}")

conn.close()

print(f"\n💡 Available for centroid calculation:")
print(f"  ✅ Euclidean distance (from existing 'distance' column)")
print(f"  ✅ Cosine similarity (from existing 'similarity' column)")
print(f"  ❌ Manhattan distance (NULL - would need recalculation)")
print(f"  ❌ Dot product (NULL - would need recalculation)")
