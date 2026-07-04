-- Schema Turso: Consolidated schema for Turso/libSQL RAG database.
--
-- Merges the two-file SQLite split (semantic-index.sqlite + embeddings.sqlite)
-- into a single Turso database with native F8_BLOB(384) vector columns and
-- DiskANN (libsql_vector_idx) ANN indexes. Uses a _schema_version table for
-- schema versioning and an _index_metadata table for application metadata,
-- both of which are read-write on Turso Cloud.
--
-- Based on schema-v2.sql with the following changes:
--   - chunk_embeddings table merged into chunks as embedding F8_BLOB(384)
--   - embedding_model TEXT, chunk_sha256 TEXT, embedded_at INTEGER added to chunks
--   - term_embeddings.embedding changed from BLOB to F8_BLOB(384)
--   - DiskANN vector indexes on chunks.embedding and term_embeddings.embedding
--   - _schema_version table for schema versioning (read-only pragmas not supported on Turso)
--   - _index_metadata table for application metadata
--   - No unsupported pragmas or cleanup commands (both unavailable on Turso Cloud)

CREATE TABLE IF NOT EXISTS documents (
  doc_id INTEGER PRIMARY KEY,
  file_path TEXT NOT NULL UNIQUE,
  doc_family TEXT NOT NULL,
  mtime_ms INTEGER NOT NULL,
  file_size INTEGER NOT NULL,
  sha256 TEXT NOT NULL,
  indexed_at INTEGER NOT NULL,
  arch_layer TEXT,
  test_coverage TEXT CHECK(test_coverage IN ('full', 'partial', 'none', 'unknown')),
  source_path_pattern TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id INTEGER PRIMARY KEY,
  doc_id INTEGER NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
  chunk_index INTEGER NOT NULL,
  heading_path TEXT,
  body_text TEXT NOT NULL,
  char_start INTEGER NOT NULL,
  char_end INTEGER NOT NULL,
  parent_chunk_id INTEGER,
  depth INTEGER NOT NULL DEFAULT 0,
  context_header TEXT,
  symbol_name TEXT,
  signature_text TEXT,
  jsdoc_text TEXT,
  export_type TEXT,
  module_path TEXT,
  arch_layer TEXT,
  jsdoc_quality TEXT CHECK(jsdoc_quality IN ('none', 'weak', 'adequate', 'good')),
  jsdoc_word_count INTEGER,
  cyclomatic_complexity INTEGER,
  test_coverage TEXT CHECK(test_coverage IN ('full', 'partial', 'none', 'unknown')),
  source_path_pattern TEXT,
  embedding F8_BLOB(384),
  embedding_model TEXT,
  chunk_sha256 TEXT,
  embedded_at INTEGER,
  UNIQUE(doc_id, chunk_index)
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  body_text,
  heading_path,
  content='chunks',
  content_rowid='chunk_id',
  tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, body_text, heading_path)
  VALUES (new.chunk_id, new.body_text, new.heading_path);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, body_text, heading_path)
  VALUES ('delete', old.chunk_id, old.body_text, old.heading_path);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, body_text, heading_path)
  VALUES ('delete', old.chunk_id, old.body_text, old.heading_path);
  INSERT INTO chunks_fts(rowid, body_text, heading_path)
  VALUES (new.chunk_id, new.body_text, new.heading_path);
END;

CREATE INDEX IF NOT EXISTS documents_family_idx ON documents(doc_family);

-- v3: Metadata filter indexes on chunks
CREATE INDEX IF NOT EXISTS chunks_arch_layer_idx ON chunks(arch_layer);
CREATE INDEX IF NOT EXISTS chunks_jsdoc_quality_idx ON chunks(jsdoc_quality);
CREATE INDEX IF NOT EXISTS chunks_test_coverage_idx ON chunks(test_coverage);
CREATE INDEX IF NOT EXISTS chunks_export_type_idx ON chunks(export_type);
CREATE INDEX IF NOT EXISTS chunks_source_path_pattern_idx ON chunks(source_path_pattern);
CREATE INDEX IF NOT EXISTS chunks_family_arch_layer_idx ON chunks(doc_id, arch_layer);
CREATE INDEX IF NOT EXISTS chunks_family_export_type_idx ON chunks(doc_id, export_type);

-- v3: Metadata filter indexes on documents
CREATE INDEX IF NOT EXISTS documents_arch_layer_idx ON documents(arch_layer);

-- v2: Semantic chunking indexes
CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS chunks_parent_idx ON chunks(parent_chunk_id);
CREATE INDEX IF NOT EXISTS chunks_depth_idx ON chunks(depth);
CREATE INDEX IF NOT EXISTS chunks_symbol_idx ON chunks(symbol_name);
CREATE INDEX IF NOT EXISTS chunks_module_idx ON chunks(module_path);

-- DiskANN vector index on chunks.embedding (metric=cosine, max_neighbors=59, alpha=1.2, search_l=80)
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks(libsql_vector_idx(embedding));

-- v4: Entity/relationship graph tables
CREATE TABLE IF NOT EXISTS entities (
  entity_id INTEGER PRIMARY KEY,
  entity_type TEXT NOT NULL CHECK(entity_type IN (
    'module', 'class', 'function', 'interface', 'type-alias', 'variable', 'error-class',
    'plan', 'skill', 'agent', 'demo', 'benchmark'
  )),
  name TEXT NOT NULL,
  qualified_name TEXT NOT NULL UNIQUE,
  doc_id INTEGER REFERENCES documents(doc_id) ON DELETE CASCADE,
  chunk_id INTEGER REFERENCES chunks(chunk_id) ON DELETE SET NULL,
  module_path TEXT,
  signature_text TEXT,
  file_path TEXT NOT NULL,
  char_start INTEGER,
  char_end INTEGER,
  extra_metadata TEXT DEFAULT '{}',
  created_at INTEGER NOT NULL DEFAULT (unixepoch())
);

CREATE INDEX IF NOT EXISTS entities_type_idx ON entities(entity_type);
CREATE INDEX IF NOT EXISTS entities_name_idx ON entities(name);
CREATE INDEX IF NOT EXISTS entities_qualified_name_idx ON entities(qualified_name);
CREATE INDEX IF NOT EXISTS entities_module_path_idx ON entities(module_path);
CREATE INDEX IF NOT EXISTS entities_doc_id_idx ON entities(doc_id);

CREATE TABLE IF NOT EXISTS edges (
  edge_id INTEGER PRIMARY KEY,
  source_entity_id INTEGER NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
  target_entity_id INTEGER NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
  relationship TEXT NOT NULL CHECK(relationship IN (
    'imports', 'exports', 'depends-on', 'implements', 'references', 'owns', 'part-of', 'contains'
  )),
  confidence TEXT NOT NULL DEFAULT 'high' CHECK(confidence IN ('high', 'medium', 'low')),
  extra_metadata TEXT DEFAULT '{}',
  created_at INTEGER NOT NULL DEFAULT (unixepoch()),
  UNIQUE(source_entity_id, target_entity_id, relationship)
);

CREATE INDEX IF NOT EXISTS edges_source_idx ON edges(source_entity_id);
CREATE INDEX IF NOT EXISTS edges_target_idx ON edges(target_entity_id);
CREATE INDEX IF NOT EXISTS edges_relationship_idx ON edges(relationship);
CREATE INDEX IF NOT EXISTS edges_source_rel_idx ON edges(source_entity_id, relationship);
CREATE INDEX IF NOT EXISTS edges_target_rel_idx ON edges(target_entity_id, relationship);

-- v5: Term embeddings for query expansion (F8_BLOB quantized)
CREATE TABLE IF NOT EXISTS term_embeddings (
  term TEXT NOT NULL,
  embedding F8_BLOB(384) NOT NULL,
  term_sha256 TEXT NOT NULL,
  model_id TEXT NOT NULL,
  model_sha256 TEXT NOT NULL,
  dimension INTEGER NOT NULL,
  frequency INTEGER NOT NULL,
  doc_family_count INTEGER NOT NULL,
  PRIMARY KEY (term, model_id)
);

CREATE INDEX IF NOT EXISTS term_embeddings_model_idx ON term_embeddings(model_id, term_sha256);
CREATE INDEX IF NOT EXISTS term_embeddings_frequency_idx ON term_embeddings(frequency DESC);

-- DiskANN vector index on term_embeddings.embedding (metric=cosine, max_neighbors=32, search_l=100)
CREATE INDEX IF NOT EXISTS term_embeddings_embedding_idx ON term_embeddings(libsql_vector_idx(embedding));

-- v6: Feedback events and scores for relevance feedback signals
CREATE TABLE IF NOT EXISTS feedback_events (
  event_id TEXT PRIMARY KEY,
  chunk_id INTEGER NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
  signal_type TEXT NOT NULL CHECK(signal_type IN ('impression', 'click', 'reference', 'positive', 'negative', 'irrelevant')),
  signal_strength REAL NOT NULL DEFAULT 0.0,
  query_hash TEXT,
  agent_id TEXT,
  context TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS feedback_events_chunk_idx ON feedback_events(chunk_id);
CREATE INDEX IF NOT EXISTS feedback_events_query_hash_idx ON feedback_events(query_hash);
CREATE INDEX IF NOT EXISTS feedback_events_created_at_idx ON feedback_events(created_at);

CREATE TABLE IF NOT EXISTS feedback_scores (
  chunk_id INTEGER PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
  total_positive INTEGER NOT NULL DEFAULT 0,
  total_negative INTEGER NOT NULL DEFAULT 0,
  total_impressions INTEGER NOT NULL DEFAULT 0,
  total_clicks INTEGER NOT NULL DEFAULT 0,
  total_references INTEGER NOT NULL DEFAULT 0,
  last_feedback_at DATETIME,
  feedback_boost REAL NOT NULL DEFAULT 0.0
);

-- Schema versioning (read-only schema pragma not supported on Turso)
CREATE TABLE IF NOT EXISTS _schema_version (
  version INTEGER NOT NULL,
  applied_at INTEGER NOT NULL
);

-- Index metadata (replaces application_id pragma)
CREATE TABLE IF NOT EXISTS _index_metadata (
  key TEXT PRIMARY KEY,
  value TEXT
);