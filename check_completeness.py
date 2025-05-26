#!/usr/bin/env python3
"""
Check which models have complete data
"""
import sqlite3

db_path = "/Users/rob/.emojifold/emojifold.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("🔍 Checking data completeness per model...")

# Get model info and their data counts
cursor.execute("""
    SELECT m.id, m.name, COUNT(sd.id) as pair_count
    FROM models m
    LEFT JOIN semantic_distances sd ON m.id = sd.model_id
    GROUP BY m.id, m.name
    ORDER BY m.id
""")

models_data = cursor.fetchall()
print("\n📊 Data per model:")
complete_models = []

for model_id, name, pair_count in models_data:
    status = "✅" if pair_count > 800000 else "❌" if pair_count == 0 else "⚠️ "
    print(f"  {model_id}: {name:<20} - {pair_count:,} pairs {status}")
    
    if pair_count > 800000:  # Complete data threshold
        complete_models.append((model_id, name))

print(f"\n🎯 Models with complete data: {len(complete_models)}")
for model_id, name in complete_models:
    print(f"  {model_id}: {name}")

# Check if we need to clean up incomplete model data
cursor.execute("SELECT COUNT(*) FROM semantic_distances WHERE model_id = 6")
gte_count = cursor.fetchone()[0]

if gte_count > 0 and gte_count < 800000:
    print(f"\n🧹 gte-large has incomplete data ({gte_count:,} pairs)")
    print("   Should we clean this up? (DELETE FROM semantic_distances WHERE model_id = 6)")

conn.close()
print(f"\n✅ Ready to calculate centroids for {len(complete_models)} complete models")
