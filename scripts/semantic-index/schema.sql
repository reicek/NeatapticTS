PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS documents (
  doc_id INTEGER PRIMARY KEY,
  file_path TEXT NOT NULL UNIQUE,
  doc_family TEXT NOT NULL,
  mtime_ms INTEGER NOT NULL,
  file_size INTEGER NOT NULL,
  sha256 TEXT NOT NULL,
  indexed_at INTEGER NOT NULL
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
CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS chunks_parent_idx ON chunks(parent_chunk_id);
CREATE INDEX IF NOT EXISTS chunks_depth_idx ON chunks(depth);
CREATE INDEX IF NOT EXISTS chunks_symbol_idx ON chunks(symbol_name);
CREATE INDEX IF NOT EXISTS chunks_module_idx ON chunks(module_path);