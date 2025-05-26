# EmojiFold: Emoji Semantic Manifold Discovery

Discover the mathematical structure underlying emoji semantics through systematic analysis of oppositional pairs.

## 🚀 Quick Start

```bash
# Initialize project
cd /Users/rob/repos/emojifold
uv venv
source .venv/bin/activate
uv pip install -e .

# Database location: ~/.emojifold/emojifold.db
# Run small test
emojifold test --pairs 50

# Run full overnight analysis
emojifold batch --model all --output results/

# Calculate model centroids
python calculate_centroids.py --db ~/.emojifold/emojifold.db
```

## 🎯 Project Goals

- **Mass computation** of semantic distances between emoji pairs
- **Discovery** of strongest oppositional dimensions in emoji space  
- **Cross-model validation** of semantic manifold structure
- **Efficient storage** and analysis of large-scale results

## 🔬 Key Research Questions

1. **Mathematical vs Visual**: How do math-oriented models compare to general embeddings?
2. **Universal Oppositions**: Which emoji pairs are strong across all models?
3. **Dimensional Structure**: What are the fundamental axes of emoji semantics?

## 📊 Expected Results

With ~3,000 emoji testing ~9 million pairs overnight, we should discover:
- Top 100 strongest oppositional pairs
- Model-specific vs universal patterns
- Natural clustering of semantic dimensions
- Validation of our ⚫⚪ and ➕➖ findings

## 🛠️ Technical Stack

- **UV** for dependency management
- **Ollama + HuggingFace** for embeddings
- **Optimized for M2 Ultra** (64GB) + optional 4090
- **SQLite** for efficient result storage
- **Async/parallel** processing for maximum throughput

---

*Ready to map the emoji semantic universe! 🌌*
