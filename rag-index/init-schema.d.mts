import type { Client } from '@libsql/client';

/** Absolute path to the repository root directory. */
export const repoRoot: string;

/** Default on-disk path for the Turso/libSQL replica database. */
export const defaultDatabasePath: string;

export interface InitSemanticIndexOptions {
  /** Existing libSQL client to use instead of opening a new database. */
  client?: Client;
  /** Path to the SQLite database file. Defaults to {@link defaultDatabasePath}. */
  databasePath?: string;
}

/**
 * Initialize the semantic index schema on a local libSQL database.
 *
 * Applies the consolidated `schema-turso.sql` and idempotently migrates A1
 * slice-metadata columns, then returns a client connected to the database.
 *
 * @param options - Configuration for the index client.
 * @returns A libSQL client with the schema applied.
 */
export function initSemanticIndex(
  options?: InitSemanticIndexOptions,
): Promise<Client>;
