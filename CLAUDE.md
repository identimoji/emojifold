# EmojiFold Project Rules & Guidelines

## 🏗️ Project Setup Standards

### Environment Configuration
- **Copy `.env.example` to `.env`** and customize for your setup
- **Edit `config.yaml`** in project root for model and processing configuration
- **Create required directories**: `./data`, `./results`, `./logs`, `./.cache`
- **Ensure Ollama is running** if using Ollama models: `ollama serve`

### Python Environment Management
- **Use UV** for all package management and installation
- **NO conda** - we stick with UV's fast, reliable dependency resolution
- **Use `.venv`** - virtual environment in project root
- **pyproject.toml** for all dependencies - NO requirements.txt files

### Model & Embedding Strategy
- **Start with Ollama** for local model serving
- **HuggingFace models** for embeddings and additional capabilities
- **HuggingFace Hub** ONLY for model downloading - no runtime dependencies
- Focus on **fast, efficient models** suitable for large-scale batch processing

### Hardware Optimization
- **Primary target**: Mac M2 Ultra with 64GB RAM
- **Secondary**: 4090 GPU for heavy lifting if needed
- Design for **overnight batch runs** - prioritize throughput over interactivity
- Leverage **parallel processing** for emoji pair distance calculations

## 🎯 Project Goals

### Core Mission
Build a comprehensive **emoji semantic manifold** by:
1. **Mass computation** of semantic distances between emoji pairs
2. **Identification** of strongest oppositional dimensions
3. **Creation** of a robust coordinate system for semantic positioning
4. **Validation** across multiple embedding models

### Technical Objectives
- **Scale**: Test thousands of emoji pairs systematically
- **Speed**: Optimize for overnight batch processing
- **Accuracy**: Cross-validate with multiple embedding models
- **Storage**: Efficient data structures for large-scale results

## 🔬 Research Questions

### Mathematical vs Visual Oppositions
- How do **math-oriented models** (e.g., specialized STEM models) compare to general models?
- Do **pure mathematical symbols** show stronger opposition in math-focused embeddings?
- What's the relationship between **training data bias** and semantic opposition strength?

### Cross-Model Calibration
- How consistent are **semantic distances** across different embedding models?
- Can we identify **universal oppositional pairs** that are strong across all models?
- What does **model-specific variation** tell us about training data differences?

### Dimensional Discovery
- What are the **strongest 50-100 emoji oppositions** across the full Unicode emoji set?
- Do oppositions cluster into **semantic categories** (visual, spatial, emotional, etc.)?
- Can we automatically discover **orthogonal dimensions** for maximum semantic coverage?

## 🚀 Implementation Strategy

### Phase 1: Foundation
- Set up UV environment with core dependencies
- Implement **semantic distance calculator** with multiple model support
- Create **batch processing pipeline** for large-scale emoji comparisons
- Design **efficient storage** for results (SQLite + optional Oxigraph)

### Phase 2: Mass Computation
- **Systematic testing** of emoji pairs across multiple embedding models
- **Parallel processing** to maximize M2 Ultra utilization
- **Progress tracking** and **resumable jobs** for overnight runs
- **Result validation** and consistency checking

### Phase 3: Analysis & Discovery
- **Statistical analysis** of semantic distance patterns
- **Clustering** and **dimensionality reduction** to find natural groupings
- **Cross-model comparison** to identify universal vs model-specific patterns
- **Visualization** of the discovered emoji manifold

## 📦 Dependency Guidelines

### Core Stack
```toml
[dependencies]
# Embedding & ML
sentence-transformers = "^2.2.2"
transformers = "^4.35.0"
torch = "^2.1.0"

# Data & Storage
pandas = "^2.1.0"
numpy = "^1.24.0"
sqlite3 = "built-in"

# Async & Performance  
asyncio = "built-in"
concurrent.futures = "built-in"
multiprocessing = "built-in"

# Optional: Advanced storage
# oxigraph = "^0.3.0"  # If we need RDF storage
```

### Model Access
- **Ollama**: Local model serving via HTTP API
- **HuggingFace**: Model downloading only - no transformers runtime if possible
- **Sentence Transformers**: Primary embedding interface
- **Custom wrappers**: For consistent model interfaces

## 🎯 Immediate Next Steps

1. **Initialize UV project** with pyproject.toml
2. **Set up basic emoji pair distance calculator**
3. **Test with small emoji set** (~50 pairs) to validate approach
4. **Scale to full emoji set** (~3000+ emojis = ~9M pairs) for overnight run
5. **Compare results** across multiple embedding models

## 💡 Optimization Notes

### For M2 Ultra
- Leverage **unified memory** for large datasets
- Use **multiprocessing** to max out CPU cores
- **Batch embed operations** for efficiency
- Consider **memory mapping** for large result sets

### For 4090 (if needed)
- **GPU acceleration** for transformer models
- **CUDA optimization** for batch operations
- **Mixed precision** to maximize throughput
- **Pipeline GPU/CPU work** to avoid bottlenecks

---

*Ready to discover the mathematical structure underlying emoji semantics! 🚀*
