-- EmojiFold Schema V3 - Unicode Codepoint Based
-- Uses Unicode codepoints as primary keys for emoji

-- Complete emoji table with codepoint as ID
CREATE TABLE emojis (
    codepoint INTEGER PRIMARY KEY,  -- Unicode codepoint (e.g., 128512 for 😀)
    emoji TEXT NOT NULL,            -- The actual emoji character
    name TEXT NOT NULL,             -- Unicode name (e.g., "GRINNING FACE")
    category TEXT NOT NULL,         -- Our category (face, animal, nature, etc.)
    unicode_block TEXT NOT NULL,    -- Unicode block name
    hex_code TEXT NOT NULL,         -- Hex representation (e.g., "U+1F600")
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Models table
CREATE TABLE models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,      -- e.g., "all-MiniLM-L6-v2"
    model_type TEXT NOT NULL,       -- 'sentence_transformer', 'ollama', 'huggingface'
    model_path TEXT NOT NULL,       -- Path or identifier for model
    embedding_dim INTEGER NOT NULL, -- Dimension of embeddings
    parameters TEXT,                -- JSON of model parameters
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Emoji pairs using codepoints as foreign keys
CREATE TABLE emoji_pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    emoji1_codepoint INTEGER NOT NULL,
    emoji2_codepoint INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (emoji1_codepoint) REFERENCES emojis (codepoint),
    FOREIGN KEY (emoji2_codepoint) REFERENCES emojis (codepoint),
    UNIQUE(emoji1_codepoint, emoji2_codepoint),
    CHECK (emoji1_codepoint < emoji2_codepoint)  -- Ensure consistent ordering
);

-- Semantic distances
CREATE TABLE semantic_distances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair_id INTEGER NOT NULL,
    model_id INTEGER NOT NULL,
    distance REAL NOT NULL,
    similarity REAL NOT NULL,
    embedding1 TEXT,                -- JSON serialized embedding (optional)
    embedding2 TEXT,                -- JSON serialized embedding (optional)
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (pair_id) REFERENCES emoji_pairs (id),
    FOREIGN KEY (model_id) REFERENCES models (id),
    UNIQUE(pair_id, model_id)
);

-- Analysis results
CREATE TABLE analysis_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER NOT NULL,
    total_pairs INTEGER NOT NULL,
    completion_time TIMESTAMP NOT NULL,
    statistics TEXT,                -- JSON serialized stats
    metadata TEXT,                  -- JSON serialized metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_id) REFERENCES models (id)
);

-- Performance indexes
CREATE INDEX idx_emojis_category ON emojis(category);
CREATE INDEX idx_emojis_block ON emojis(unicode_block);
CREATE INDEX idx_emoji_pairs_codepoints ON emoji_pairs(emoji1_codepoint, emoji2_codepoint);
CREATE INDEX idx_semantic_distances_model ON semantic_distances(model_id);
CREATE INDEX idx_semantic_distances_distance ON semantic_distances(distance DESC);
CREATE INDEX idx_semantic_distances_pair ON semantic_distances(pair_id);

-- Convenient views for queries
CREATE VIEW emoji_distance_view AS
SELECT 
    e1.emoji as emoji1,
    e1.name as name1,
    e1.category as category1,
    e1.codepoint as codepoint1,
    e2.emoji as emoji2,
    e2.name as name2,
    e2.category as category2, 
    e2.codepoint as codepoint2,
    m.name as model_name,
    sd.distance,
    sd.similarity,
    sd.calculated_at
FROM semantic_distances sd
JOIN emoji_pairs ep ON sd.pair_id = ep.id
JOIN emojis e1 ON ep.emoji1_codepoint = e1.codepoint
JOIN emojis e2 ON ep.emoji2_codepoint = e2.codepoint
JOIN models m ON sd.model_id = m.id;

-- View for strongest oppositions by category
CREATE VIEW strongest_oppositions_by_category AS
SELECT 
    e1.category as category1,
    e2.category as category2,
    e1.emoji as emoji1,
    e2.emoji as emoji2,
    m.name as model_name,
    sd.distance,
    ROW_NUMBER() OVER (
        PARTITION BY e1.category, e2.category, m.name 
        ORDER BY sd.distance DESC
    ) as rank_in_category
FROM semantic_distances sd
JOIN emoji_pairs ep ON sd.pair_id = ep.id
JOIN emojis e1 ON ep.emoji1_codepoint = e1.codepoint
JOIN emojis e2 ON ep.emoji2_codepoint = e2.codepoint
JOIN models m ON sd.model_id = m.id;

-- View for cross-category analysis
CREATE VIEW cross_category_distances AS
SELECT 
    e1.category as category1,
    e2.category as category2,
    COUNT(*) as pair_count,
    AVG(sd.distance) as avg_distance,
    MAX(sd.distance) as max_distance,
    MIN(sd.distance) as min_distance,
    m.name as model_name
FROM semantic_distances sd
JOIN emoji_pairs ep ON sd.pair_id = ep.id
JOIN emojis e1 ON ep.emoji1_codepoint = e1.codepoint
JOIN emojis e2 ON ep.emoji2_codepoint = e2.codepoint
JOIN models m ON sd.model_id = m.id
GROUP BY e1.category, e2.category, m.name;
