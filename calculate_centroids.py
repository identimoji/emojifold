#!/usr/bin/env python3
"""
Emojifold Centroid Calculator
Calculates model centroids, universal centroids, and model deviations for emojifold-1337
"""

import sqlite3
import json
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime
import argparse

def parse_embedding(embedding_str: str) -> np.ndarray:
    """Parse JSON embedding string to numpy array"""
    return np.array(json.loads(embedding_str))

def calculate_euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate Euclidean distance between two vectors"""
    return np.linalg.norm(vec1 - vec2)

def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot_product / (norm_a * norm_b)

def calculate_manhattan_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate Manhattan (L1) distance between two vectors"""
    return np.sum(np.abs(vec1 - vec2))

def calculate_dot_product(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate dot product between two vectors"""
    return np.dot(vec1, vec2)

class EmojifoldCentroidCalculator:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        
    def get_models(self) -> List[Dict]:
        """Get all models from database"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM models ORDER BY id")
        return [dict(row) for row in cursor.fetchall()]
    
    def get_all_embeddings_for_model(self, model_id: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get all embedding pairs for a specific model"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT embedding1, embedding2 
            FROM semantic_distances 
            WHERE model_id = ? AND embedding1 IS NOT NULL AND embedding2 IS NOT NULL
        """, (model_id,))
        
        embeddings = []
        for row in cursor.fetchall():
            emb1 = parse_embedding(row['embedding1'])
            emb2 = parse_embedding(row['embedding2'])
            embeddings.extend([emb1, emb2])
        
        return embeddings
    
    def calculate_model_centroid(self, model_id: int, metric_type: str = 'euclidean') -> Dict[str, Any]:
        """Calculate centroid and extrema stats for a model"""
        print(f"Calculating centroid for model {model_id}...")
        
        embeddings = self.get_all_embeddings_for_model(model_id)
        if not embeddings:
            raise ValueError(f"No embeddings found for model {model_id}")
        
        # Convert to numpy array for efficient computation
        embedding_matrix = np.array(embeddings)
        
        # Calculate centroid (center of mass)
        centroid = np.mean(embedding_matrix, axis=0)
        
        # Calculate extrema stats (moment of inertia equivalent)
        min_vals = np.min(embedding_matrix, axis=0)
        max_vals = np.max(embedding_matrix, axis=0)
        std_vals = np.std(embedding_matrix, axis=0)
        var_vals = np.var(embedding_matrix, axis=0)
        
        # Calculate distances from centroid for inertia-like measure
        distances_from_centroid = np.array([
            np.linalg.norm(emb - centroid) for emb in embedding_matrix
        ])
        
        extrema_stats = {
            'min_values': min_vals.tolist(),
            'max_values': max_vals.tolist(),
            'std_values': std_vals.tolist(),
            'var_values': var_vals.tolist(),
            'mean_distance_from_centroid': float(np.mean(distances_from_centroid)),
            'std_distance_from_centroid': float(np.std(distances_from_centroid)),
            'max_distance_from_centroid': float(np.max(distances_from_centroid)),
            'total_embeddings': len(embeddings),
            'embedding_dimension': len(centroid)
        }
        
        return {
            'centroid': centroid.tolist(),
            'extrema_stats': extrema_stats
        }
    
    def store_model_centroid(self, model_id: int, metric_type: str, centroid_data: Dict, version: int = 1):
        """Store model centroid in database"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO model_centroids 
            (model_id, metric_type, centroid_vector, extrema_stats, version, calculated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            model_id,
            metric_type,
            json.dumps(centroid_data['centroid']),
            json.dumps(centroid_data['extrema_stats']),
            version,
            datetime.now().isoformat()
        ))
        self.conn.commit()
    
    def calculate_universal_centroid(self, metric_type: str = 'euclidean', version: int = 1) -> Dict[str, Any]:
        """Calculate centroid of centroids (universal centroid)"""
        print(f"Calculating universal centroid for metric: {metric_type}")
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT mc.model_id, mc.centroid_vector, m.name
            FROM model_centroids mc
            JOIN models m ON mc.model_id = m.id
            WHERE mc.metric_type = ? AND mc.version = ?
            ORDER BY mc.model_id
        """, (metric_type, version))
        
        model_centroids = []
        model_ids = []
        model_names = []
        
        for row in cursor.fetchall():
            centroid = json.loads(row['centroid_vector'])
            model_centroids.append(centroid)
            model_ids.append(row['model_id'])
            model_names.append(row['name'])
        
        if not model_centroids:
            raise ValueError(f"No model centroids found for metric {metric_type}")
        
        # Calculate universal centroid (centroid of centroids)
        centroid_matrix = np.array(model_centroids)
        universal_centroid = np.mean(centroid_matrix, axis=0)
        
        # Calculate stability metrics
        centroid_distances = []
        for i, centroid in enumerate(model_centroids):
            dist = np.linalg.norm(np.array(centroid) - universal_centroid)
            centroid_distances.append(dist)
        
        stability_metrics = {
            'mean_deviation_from_universal': float(np.mean(centroid_distances)),
            'std_deviation_from_universal': float(np.std(centroid_distances)),
            'max_deviation_from_universal': float(np.max(centroid_distances)),
            'min_deviation_from_universal': float(np.min(centroid_distances)),
            'model_names': model_names,
            'individual_deviations': dict(zip(model_names, centroid_distances))
        }
        
        return {
            'universal_centroid': universal_centroid.tolist(),
            'stability_metrics': stability_metrics,
            'model_ids': model_ids
        }
    
    def store_universal_centroid(self, metric_type: str, universal_data: Dict, version: int = 1):
        """Store universal centroid in database"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO universal_centroids
            (metric_type, model_set_version, model_ids, universal_centroid, stability_metrics, calculated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            metric_type,
            version,
            json.dumps(universal_data['model_ids']),
            json.dumps(universal_data['universal_centroid']),
            json.dumps(universal_data['stability_metrics']),
            datetime.now().isoformat()
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def calculate_model_deviations(self, universal_centroid_id: int):
        """Calculate and store model deviations from universal centroid"""
        print("Calculating model deviations from universal centroid...")
        
        # Get universal centroid data
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM universal_centroids WHERE id = ?
        """, (universal_centroid_id,))
        
        universal_row = cursor.fetchone()
        if not universal_row:
            raise ValueError(f"Universal centroid {universal_centroid_id} not found")
        
        universal_centroid = np.array(json.loads(universal_row['universal_centroid']))
        metric_type = universal_row['metric_type']
        version = universal_row['model_set_version']
        
        # Get model centroids for this metric and version
        cursor.execute("""
            SELECT mc.*, m.name as model_name
            FROM model_centroids mc
            JOIN models m ON mc.model_id = m.id
            WHERE mc.metric_type = ? AND mc.version = ?
        """, (metric_type, version))
        
        for row in cursor.fetchall():
            model_centroid = np.array(json.loads(row['centroid_vector']))
            deviation_vector = model_centroid - universal_centroid
            deviation_magnitude = np.linalg.norm(deviation_vector)
            
            # Create personality signature (interpretable traits)
            extrema_stats = json.loads(row['extrema_stats'])
            personality_signature = {
                'model_name': row['model_name'],
                'deviation_direction': 'positive' if np.mean(deviation_vector) > 0 else 'negative',
                'max_deviation_dimension': int(np.argmax(np.abs(deviation_vector))),
                'spread_from_universal': float(deviation_magnitude),
                'relative_spread_rank': None,  # Will be filled in later
                'embedding_space_size': extrema_stats['mean_distance_from_centroid']
            }
            
            # Store deviation
            cursor.execute("""
                INSERT OR REPLACE INTO model_deviations
                (model_id, universal_centroid_id, deviation_vector, deviation_magnitude, personality_signature, calculated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                row['model_id'],
                universal_centroid_id,
                json.dumps(deviation_vector.tolist()),
                deviation_magnitude,
                json.dumps(personality_signature),
                datetime.now().isoformat()
            ))
        
        self.conn.commit()
    
    def run_full_analysis(self, metrics: List[str] = None, version: int = 1):
        """Run complete centroid analysis"""
        if metrics is None:
            metrics = ['euclidean']  # Start with euclidean, add others later
        
        models = self.get_models()
        print(f"Found {len(models)} models")
        
        for metric in metrics:
            print(f"\n=== Processing metric: {metric} ===")
            
            # Step 1: Calculate model centroids
            for model in models:
                try:
                    centroid_data = self.calculate_model_centroid(model['id'], metric)
                    self.store_model_centroid(model['id'], metric, centroid_data, version)
                    print(f"✓ Model {model['name']} centroid calculated")
                except Exception as e:
                    print(f"✗ Error calculating centroid for model {model['name']}: {e}")
            
            # Step 2: Calculate universal centroid
            try:
                universal_data = self.calculate_universal_centroid(metric, version)
                universal_id = self.store_universal_centroid(metric, universal_data, version)
                print(f"✓ Universal centroid calculated (ID: {universal_id})")
                
                # Step 3: Calculate model deviations
                self.calculate_model_deviations(universal_id)
                print(f"✓ Model deviations calculated")
                
            except Exception as e:
                print(f"✗ Error calculating universal centroid for {metric}: {e}")
    
    def print_summary(self, metric_type: str = 'euclidean', version: int = 1):
        """Print analysis summary"""
        cursor = self.conn.cursor()
        
        # Get universal centroid info
        cursor.execute("""
            SELECT uc.*, 
                   COUNT(md.id) as deviation_count
            FROM universal_centroids uc
            LEFT JOIN model_deviations md ON uc.id = md.universal_centroid_id
            WHERE uc.metric_type = ? AND uc.model_set_version = ?
            GROUP BY uc.id
        """, (metric_type, version))
        
        uc_row = cursor.fetchone()
        if not uc_row:
            print(f"No analysis found for {metric_type} metric version {version}")
            return
        
        stability = json.loads(uc_row['stability_metrics'])
        
        print(f"\n=== EMOJIFOLD-1337 ANALYSIS SUMMARY ({metric_type.upper()}) ===")
        print(f"Models analyzed: {len(stability['model_names'])}")
        print(f"Model names: {', '.join(stability['model_names'])}")
        print(f"Universal centroid stability:")
        print(f"  Mean deviation: {stability['mean_deviation_from_universal']:.6f}")
        print(f"  Std deviation: {stability['std_deviation_from_universal']:.6f}")
        print(f"  Max deviation: {stability['max_deviation_from_universal']:.6f}")
        
        print(f"\nModel personality signatures:")
        for model_name, deviation in stability['individual_deviations'].items():
            print(f"  {model_name}: {deviation:.6f} units from universal center")
    
    def close(self):
        """Close database connection"""
        self.conn.close()

def main():
    parser = argparse.ArgumentParser(description='Calculate emojifold centroids')
    parser.add_argument('--db', default='/Users/rob/.emojifold/emojifold.db', help='Database path')
    parser.add_argument('--metrics', nargs='+', default=['euclidean'], 
                       choices=['euclidean', 'cosine', 'manhattan', 'dot_product'],
                       help='Metrics to calculate')
    parser.add_argument('--version', type=int, default=1, help='Model set version')
    parser.add_argument('--summary-only', action='store_true', help='Only print summary')
    
    args = parser.parse_args()
    
    calculator = EmojifoldCentroidCalculator(args.db)
    
    try:
        if not args.summary_only:
            calculator.run_full_analysis(args.metrics, args.version)
        
        for metric in args.metrics:
            calculator.print_summary(metric, args.version)
            
    finally:
        calculator.close()

if __name__ == '__main__':
    main()
