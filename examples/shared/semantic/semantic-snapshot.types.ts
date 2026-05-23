/**
 * A single markdown chunk from the generated semantic browser snapshot.
 *
 * @example
 * ```ts
 * const heading = chunk.heading_path;
 * const text = chunk.body_text;
 * ```
 */
export interface SemanticSnapshotChunk {
  /** Stable chunk primary key from the SQLite corpus index. */
  chunk_id: number;
  /** Markdown heading trail that gives the chunk its local documentation context. */
  heading_path: string;
  /** Plain markdown body text for lightweight browser-side lexical search. */
  body_text: string;
  /** Inclusive character offset in the source document. */
  char_start: number;
  /** Exclusive character offset in the source document. */
  char_end: number;
}

/**
 * A repository document and its ordered chunks in the browser snapshot.
 */
export interface SemanticSnapshotDocument {
  /** Stable document primary key from the SQLite corpus index. */
  doc_id: number;
  /** Repository-relative path; absolute local paths are intentionally excluded. */
  file_path: string;
  /** Corpus family such as readme, plan, demo, skill, or agent. */
  family: string;
  /** Chunks ordered by their source-document position. */
  chunks: SemanticSnapshotChunk[];
}

/**
 * Browser-friendly static export of the Repo Cortex semantic corpus.
 */
export interface SemanticSnapshot {
  /** Snapshot schema version used by browser loaders to reject incompatible payloads. */
  schema_version: '1';
  /** ISO timestamp from the generation run; also acts as the cache freshness key. */
  generated_at: string;
  /** Sorted list of corpus families represented in the snapshot. */
  families: string[];
  /** Repository documents and nested chunks in deterministic path order. */
  documents: SemanticSnapshotDocument[];
}

/** Options for loading and caching the browser snapshot. */
export interface LoadSemanticSnapshotOptions {
  /** IndexedDB database name used by browser demos. */
  databaseName?: string;
  /** IndexedDB object store name used for generated_at keyed snapshots. */
  storeName?: string;
  /** Set true to bypass cache lookup and fetch the served snapshot. */
  forceRefresh?: boolean;
}

/** Controls how many lexical matches are returned from a snapshot query. */
export interface SearchSnapshotOptions {
  /** Maximum number of ranked chunk matches to return. */
  limit?: number;
}

/** A single ranked chunk match from browser-side semantic snapshot search. */
export interface SemanticSnapshotSearchResult {
  /** Heading-weighted lexical score; higher values rank first. */
  score: number;
  /** Matched repository document. */
  document: SemanticSnapshotDocument;
  /** Matched chunk inside the document. */
  chunk: SemanticSnapshotChunk;
}
