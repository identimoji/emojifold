#!/usr/bin/env python3
"""
Quick database connection test
"""
import sqlite3
import os

db_path = "/Users/rob/.emojifold/emojifold.db"

print(f"🔍 Checking database at: {db_path}")
print(f"📁 Database exists: {os.path.exists(db_path)}")

if os.path.exists(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"📊 Tables found: {tables}")
        
        # Check models
        if 'models' in tables:
            cursor.execute("SELECT COUNT(*) as count FROM models")
            model_count = cursor.fetchone()[0]
            print(f"🤖 Models in database: {model_count}")
            
            cursor.execute("SELECT id, name FROM models")
            models = cursor.fetchall()
            for model_id, name in models:
                print(f"  {model_id}: {name}")
        
        # Check semantic distances
        if 'semantic_distances' in tables:
            cursor.execute("SELECT COUNT(*) as count FROM semantic_distances")
            distance_count = cursor.fetchone()[0]
            print(f"📏 Semantic distances: {distance_count:,}")
        
        print("✅ Database connection successful!")
        conn.close()
        
    except Exception as e:
        print(f"❌ Database error: {e}")
else:
    print("❌ Database file not found!")
    print("Expected location: ~/.emojifold/emojifold.db")
