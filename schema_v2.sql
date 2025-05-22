-- Better normalized schema for EmojiFold

-- Individual emojis with unique IDs
CREATE TABLE emojis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    emoji TEXT UNIQUE NOT NULL,
    description TEXT,
    category TEXT,  -- 'face', 'animal', 'symbol', 'nature', 'color'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Models table
CREATE TABLE models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    model_type TEXT NOT NULL,  -- 'sentence_transformer', 'ollama', 'huggingface'
    model_path TEXT NOT NULL,
    embedding_dim INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Emoji pairs with foreign keys to emojis table
CREATE TABLE emoji_pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    emoji1_id INTEGER NOT NULL,
    emoji2_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (emoji1_id) REFERENCES emojis (id),
    FOREIGN KEY (emoji2_id) REFERENCES emojis (id),
    UNIQUE(emoji1_id, emoji2_id),
    CHECK (emoji1_id < emoji2_id)  -- Ensure consistent ordering
);

-- Semantic distances with foreign keys
CREATE TABLE semantic_distances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair_id INTEGER NOT NULL,
    model_id INTEGER NOT NULL,
    distance REAL NOT NULL,
    similarity REAL NOT NULL,
    embedding1 TEXT,  -- JSON serialized embedding (optional)
    embedding2 TEXT,  -- JSON serialized embedding (optional)
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (pair_id) REFERENCES emoji_pairs (id),
    FOREIGN KEY (model_id) REFERENCES models (id),
    UNIQUE(pair_id, model_id)
);

-- Analysis results linked to model
CREATE TABLE analysis_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER NOT NULL,
    total_pairs INTEGER NOT NULL,
    completion_time TIMESTAMP NOT NULL,
    statistics TEXT,  -- JSON serialized stats
    metadata TEXT,    -- JSON serialized metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_id) REFERENCES models (id)
);

-- Indexes for performance
CREATE INDEX idx_semantic_distances_model ON semantic_distances(model_id);
CREATE INDEX idx_semantic_distances_distance ON semantic_distances(distance DESC);
CREATE INDEX idx_emoji_pairs_emojis ON emoji_pairs(emoji1_id, emoji2_id);
CREATE INDEX idx_emojis_category ON emojis(category);

-- Views for easy querying
CREATE VIEW emoji_distances_view AS
SELECT 
    e1.emoji as emoji1,
    e1.description as description1,
    e1.category as category1,
    e2.emoji as emoji2,
    e2.description as description2,
    e2.category as category2,
    m.name as model_name,
    sd.distance,
    sd.similarity,
    sd.calculated_at
FROM semantic_distances sd
JOIN emoji_pairs ep ON sd.pair_id = ep.id
JOIN emojis e1 ON ep.emoji1_id = e1.id  
JOIN emojis e2 ON ep.emoji2_id = e2.id
JOIN models m ON sd.model_id = m.id;
