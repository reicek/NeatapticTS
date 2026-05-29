import Database from 'better-sqlite3';
import { tokenizeNeatChatText } from '../core/neatChat.tokenization.utils';
import {
  DEFAULT_DB_PATH,
  MEMORY_BM25_B,
  MEMORY_BM25_K1,
} from './neatChat.memory.constants';
import { sanitizeFtsQuery } from './neatChat.memory.fts';
import type {
  CreateStoredMemoryEntry,
  MemoryAdapter,
  MemoryResult,
  MemorySearchQuery,
  StoredMemoryEntry,
} from './neatChat.memory.types';

const MEMORY_SCHEMA_SQL = `
CREATE TABLE IF NOT EXISTS memory_entries (
  entry_id    INTEGER PRIMARY KEY,
  session_id  TEXT NOT NULL,
  entry_type  TEXT NOT NULL,
  content     TEXT NOT NULL,
  tokens      TEXT,
  score       REAL DEFAULT 1.0,
  created_at  INTEGER NOT NULL,
  last_used   INTEGER NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
  content,
  tokens,
  content='memory_entries',
  content_rowid='entry_id',
  tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS memory_fts_ai AFTER INSERT ON memory_entries BEGIN
  INSERT INTO memory_fts(rowid, content, tokens) VALUES (new.entry_id, new.content, new.tokens);
END;

CREATE TRIGGER IF NOT EXISTS memory_fts_ad AFTER DELETE ON memory_entries BEGIN
  INSERT INTO memory_fts(memory_fts, rowid, content, tokens)
    VALUES ('delete', old.entry_id, old.content, old.tokens);
END;

CREATE TRIGGER IF NOT EXISTS memory_fts_au AFTER UPDATE ON memory_entries BEGIN
  INSERT INTO memory_fts(memory_fts, rowid, content, tokens)
    VALUES ('delete', old.entry_id, old.content, old.tokens);
  INSERT INTO memory_fts(rowid, content, tokens) VALUES (new.entry_id, new.content, new.tokens);
END;
`;

type SqliteListRow = {
  readonly entry_id: number;
  readonly session_id: string;
  readonly entry_type: StoredMemoryEntry['entryType'];
  readonly content: string;
  readonly tokens: string | null;
  readonly score: number;
  readonly created_at: number;
  readonly last_used: number;
};

type SqliteSearchRow = SqliteListRow & {
  readonly bm25_score: number;
};

/** Options for creating the Node-side durable SQLite adapter. */
export interface CreateSqliteMemoryAdapterOptions {
  /** Optional database path override used by tests or alternate runtimes. */
  readonly databasePath?: string;
}

/**
 * SQLite-backed durable memory adapter for the Node runtime.
 *
 * The adapter owns the durable schema, the FTS5 content table, and the sync
 * triggers that keep ranked search aligned with the base table.
 */
export class SqliteMemoryAdapter implements MemoryAdapter {
  private readonly database: Database.Database;

  public constructor(options: CreateSqliteMemoryAdapterOptions = {}) {
    this.database = new Database(options.databasePath ?? DEFAULT_DB_PATH);
    this.database.exec(MEMORY_SCHEMA_SQL);
  }

  /** @inheritdoc */
  public async store(entry: CreateStoredMemoryEntry): Promise<string> {
    const insertResult = this.database
      .prepare(
        `
          INSERT INTO memory_entries (
            session_id,
            entry_type,
            content,
            tokens,
            score,
            created_at,
            last_used
          ) VALUES (?, ?, ?, ?, ?, ?, ?)
        `,
      )
      .run(
        entry.sessionId,
        entry.entryType,
        entry.content,
        entry.tokens.join(' '),
        entry.score,
        entry.createdAt,
        entry.lastUsed,
      );

    return String(insertResult.lastInsertRowid);
  }

  /** @inheritdoc */
  public async list(sessionId: string): Promise<readonly StoredMemoryEntry[]> {
    const rows = this.database
      .prepare(
        `
          SELECT
            entry_id,
            session_id,
            entry_type,
            content,
            tokens,
            score,
            created_at,
            last_used
          FROM memory_entries
          WHERE session_id = ?
          ORDER BY created_at ASC
        `,
      )
      .all(sessionId) as readonly SqliteListRow[];

    return rows.map(mapSqliteRowToStoredMemoryEntry);
  }

  /** @inheritdoc */
  public async remove(entryIds: readonly string[]): Promise<number> {
    if (entryIds.length === 0) {
      return 0;
    }

    const placeholders = entryIds.map(() => '?').join(', ');
    const deletionResult = this.database
      .prepare(`DELETE FROM memory_entries WHERE entry_id IN (${placeholders})`)
      .run(...entryIds);

    return deletionResult.changes;
  }

  /** @inheritdoc */
  public async search(
    query: MemorySearchQuery,
  ): Promise<readonly MemoryResult[]> {
    const sanitizedQuery = sanitizeFtsQuery(query.query);

    if (sanitizedQuery.length === 0) {
      return [];
    }

    const rows = this.database
      .prepare(
        `
          SELECT
            memory_entries.entry_id,
            memory_entries.session_id,
            memory_entries.entry_type,
            memory_entries.content,
            memory_entries.tokens,
            memory_entries.score,
            memory_entries.created_at,
            memory_entries.last_used,
            bm25(memory_fts, ${MEMORY_BM25_B}, ${MEMORY_BM25_K1}) AS bm25_score
          FROM memory_fts
          JOIN memory_entries ON memory_entries.entry_id = memory_fts.rowid
          WHERE memory_entries.session_id = ?
            AND memory_fts MATCH ?
          ORDER BY bm25_score ASC, memory_entries.last_used DESC
          LIMIT ?
        `,
      )
      .all(
        query.sessionId,
        sanitizedQuery,
        query.maxResults,
      ) as readonly SqliteSearchRow[];
    const queryTokens = tokenizeNeatChatText(
      sanitizedQuery,
      Number.MAX_SAFE_INTEGER,
    );

    return rows.map((row) => {
      const storedEntry = mapSqliteRowToStoredMemoryEntry(row);
      const overlapScore = queryTokens.reduce(
        (matchedTokenCount, queryToken) =>
          matchedTokenCount + (storedEntry.tokens.includes(queryToken) ? 1 : 0),
        0,
      );
      const normalizedBm25 = 1 / (1 + Math.abs(row.bm25_score));

      return {
        ...storedEntry,
        overlapScore,
        bm25Score: Number(normalizedBm25.toFixed(6)),
        relevanceScore: Number((overlapScore + normalizedBm25).toFixed(6)),
      } satisfies MemoryResult;
    });
  }

  /** @inheritdoc */
  public async close(): Promise<void> {
    this.database.close();
  }
}

function mapSqliteRowToStoredMemoryEntry(
  row: SqliteListRow,
): StoredMemoryEntry {
  return {
    entryId: String(row.entry_id),
    sessionId: row.session_id,
    entryType: row.entry_type,
    content: row.content,
    tokens: row.tokens?.split(/\s+/u).filter((token) => token.length > 0) ?? [],
    score: row.score,
    createdAt: row.created_at,
    lastUsed: row.last_used,
  };
}
