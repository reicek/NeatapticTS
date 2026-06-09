/**
 * @description Schema migration from v1 to v2. Adds semantic chunking columns
 * (parent_chunk_id, depth, context_header, symbol_name, signature_text, jsdoc_text,
 * export_type, module_path) to the chunks table with backward-compatible defaults.
 * Existing data is preserved — new columns receive NULL or 0 defaults so that
 * existing queries continue to work without modification.
 *
 * After running this migration, run `node scripts/semantic-index/build-index.mjs --force`
 * to re-chunk the corpus with v2 chunkers, then `node scripts/semantic-index/embed-index.mjs`
 * to re-embed all chunks (chunk SHA-256 changes because context_header and new metadata
 * are included in the hash).
 *
 * @param {string} [databasePath] - Path to the SQLite database file.
 * @param {boolean} [--json] - Emit JSON result on success.
 * @param {boolean} [--help] - Show help and exit.
 *
 * @returns {void} Exits 0 on success, 1 on error.
 *
 * @example
 * node scripts/semantic-index/migrate-schema.mjs --json
 */
import Database from 'better-sqlite3';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import {
  fail,
  parseCliArgs,
  printHelp,
  writeJsonOrText,
} from './cli-utils.mjs';
import { defaultDatabasePath, repoRoot } from './init-schema.mjs';

/** Columns to add to the chunks table for v2 semantic chunking. */
const V2_COLUMNS = [
  { name: 'parent_chunk_id', type: 'INTEGER', default: null },
  { name: 'depth', type: 'INTEGER NOT NULL', default: 0 },
  { name: 'context_header', type: 'TEXT', default: null },
  { name: 'symbol_name', type: 'TEXT', default: null },
  { name: 'signature_text', type: 'TEXT', default: null },
  { name: 'jsdoc_text', type: 'TEXT', default: null },
  { name: 'export_type', type: 'TEXT', default: null },
  { name: 'module_path', type: 'TEXT', default: null },
];

/** Columns to add to the chunks table for v3 metadata enrichment. */
const V3_CHUNK_COLUMNS = [
  { name: 'arch_layer', type: 'TEXT', default: null },
  { name: 'jsdoc_quality', type: 'TEXT', default: null },
  { name: 'jsdoc_word_count', type: 'INTEGER', default: null },
  { name: 'cyclomatic_complexity', type: 'INTEGER', default: null },
  { name: 'test_coverage', type: 'TEXT', default: null },
  { name: 'source_path_pattern', type: 'TEXT', default: null },
];

/** Columns to add to the documents table for v3 metadata enrichment. */
const V3_DOCUMENT_COLUMNS = [
  { name: 'arch_layer', type: 'TEXT', default: null },
  { name: 'test_coverage', type: 'TEXT', default: null },
  { name: 'source_path_pattern', type: 'TEXT', default: null },
];

/** Indexes to create for v3 metadata enrichment. */
const V3_INDEXES = [
  {
    name: 'chunks_arch_layer_idx',
    sql: 'CREATE INDEX IF NOT EXISTS chunks_arch_layer_idx ON chunks(arch_layer)',
  },
  {
    name: 'chunks_jsdoc_quality_idx',
    sql: 'CREATE INDEX IF NOT EXISTS chunks_jsdoc_quality_idx ON chunks(jsdoc_quality)',
  },
  {
    name: 'chunks_test_coverage_idx',
    sql: 'CREATE INDEX IF NOT EXISTS chunks_test_coverage_idx ON chunks(test_coverage)',
  },
  {
    name: 'chunks_export_type_idx',
    sql: 'CREATE INDEX IF NOT EXISTS chunks_export_type_idx ON chunks(export_type)',
  },
  {
    name: 'chunks_source_path_pattern_idx',
    sql: 'CREATE INDEX IF NOT EXISTS chunks_source_path_pattern_idx ON chunks(source_path_pattern)',
  },
  {
    name: 'chunks_family_arch_layer_idx',
    sql: 'CREATE INDEX IF NOT EXISTS chunks_family_arch_layer_idx ON chunks(doc_id, arch_layer)',
  },
  {
    name: 'chunks_family_export_type_idx',
    sql: 'CREATE INDEX IF NOT EXISTS chunks_family_export_type_idx ON chunks(doc_id, export_type)',
  },
  {
    name: 'documents_arch_layer_idx',
    sql: 'CREATE INDEX IF NOT EXISTS documents_arch_layer_idx ON documents(arch_layer)',
  },
];

/**
 * Migrate the semantic index database from v1 schema to v2 schema.
 *
 * Adds all v2 columns to the chunks table if they do not already exist.
 * Creates v2 indexes (parent_chunk_id, depth, symbol_name, module_path)
 * if they do not already exist. This migration is idempotent — running it
 * on an already-migrated database is safe and produces no changes.
 *
 * @param {object} [options={}] - Migration options.
 * @param {string} [options.databasePath] - Override database path.
 * @returns {{ migrated: string[], skipped: string[], indexes_created: string[] }} Migration result.
 * @throws {Error} When the database file does not exist.
 */
export function migrateSchemaV1ToV2(options = {}) {
  const databasePath = path.resolve(
    options.databasePath ?? defaultDatabasePath,
  );
  if (!existsSync(databasePath)) {
    throw new Error(
      `Database not found: ${databasePath}. Run build-index.mjs first.`,
    );
  }

  const database = new Database(databasePath);
  const migrated = [];
  const skipped = [];
  const indexesCreated = [];

  try {
    // Step 1: Detect existing columns in the chunks table.
    const existingColumns = new Set(
      database
        .prepare('PRAGMA table_info(chunks)')
        .all()
        .map((column) => column.name),
    );

    // Step 2: Add each v2 column if it does not already exist.
    for (const column of V2_COLUMNS) {
      if (existingColumns.has(column.name)) {
        skipped.push(column.name);
        continue;
      }

      const defaultClause =
        column.default === null
          ? ''
          : ` DEFAULT ${typeof column.default === 'number' ? column.default : `'${column.default}'`}`;

      database.exec(
        `ALTER TABLE chunks ADD COLUMN ${column.name} ${column.type}${defaultClause}`,
      );
      migrated.push(column.name);
    }

    // Step 3: Create v2 indexes if they do not already exist.
    const v2Indexes = [
      {
        name: 'chunks_parent_idx',
        sql: 'CREATE INDEX IF NOT EXISTS chunks_parent_idx ON chunks(parent_chunk_id)',
      },
      {
        name: 'chunks_depth_idx',
        sql: 'CREATE INDEX IF NOT EXISTS chunks_depth_idx ON chunks(depth)',
      },
      {
        name: 'chunks_symbol_idx',
        sql: 'CREATE INDEX IF NOT EXISTS chunks_symbol_idx ON chunks(symbol_name)',
      },
      {
        name: 'chunks_module_idx',
        sql: 'CREATE INDEX IF NOT EXISTS chunks_module_idx ON chunks(module_path)',
      },
    ];

    const existingIndexes = new Set(
      database
        .prepare(
          "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='chunks'",
        )
        .all()
        .map((row) => row.name),
    );

    for (const indexDef of v2Indexes) {
      if (!existingIndexes.has(indexDef.name)) {
        database.exec(indexDef.sql);
        indexesCreated.push(indexDef.name);
      }
    }
  } finally {
    database.close();
  }

  return { indexes_created: indexesCreated, migrated, skipped };
}

/**
 * Migrate the semantic index database from v2 to v3 schema.
 *
 * Adds metadata enrichment columns to both `chunks` and `documents` tables:
 * - chunks: arch_layer, jsdoc_quality, jsdoc_word_count, cyclomatic_complexity,
 *   test_coverage, source_path_pattern
 * - documents: arch_layer, test_coverage, source_path_pattern
 *
 * Creates v3 filter indexes if they do not already exist. This migration is
 * idempotent — running it on an already-migrated database is safe and
 * produces no changes.
 *
 * After running this migration, run `node scripts/semantic-index/build-index.mjs --force`
 * to re-enrich all chunks and documents with v3 metadata columns.
 *
 * @param {object} [options={}] - Migration options.
 * @param {string} [options.databasePath] - Override database path.
 * @returns {{ migrated_chunks: string[], skipped_chunks: string[], migrated_documents: string[], skipped_documents: string[], indexes_created: string[] }} Migration result.
 * @throws {Error} When the database file does not exist.
 */
export function migrateSchemaV2ToV3(options = {}) {
  const databasePath = path.resolve(
    options.databasePath ?? defaultDatabasePath,
  );
  if (!existsSync(databasePath)) {
    throw new Error(
      `Database not found: ${databasePath}. Run build-index.mjs first.`,
    );
  }

  const database = new Database(databasePath);
  const migratedChunks = [];
  const skippedChunks = [];
  const migratedDocuments = [];
  const skippedDocuments = [];
  const indexesCreated = [];

  try {
    // Step 1: Detect existing columns in the chunks table.
    const existingChunkColumns = new Set(
      database
        .prepare('PRAGMA table_info(chunks)')
        .all()
        .map((column) => column.name),
    );

    // Step 2: Add each v3 chunk column if it does not already exist.
    for (const column of V3_CHUNK_COLUMNS) {
      if (existingChunkColumns.has(column.name)) {
        skippedChunks.push(column.name);
        continue;
      }
      const defaultClause =
        column.default === null ? '' : ` DEFAULT ${column.default}`;
      database.exec(
        `ALTER TABLE chunks ADD COLUMN ${column.name} ${column.type}${defaultClause}`,
      );
      migratedChunks.push(column.name);
    }

    // Step 3: Detect existing columns in the documents table.
    const existingDocumentColumns = new Set(
      database
        .prepare('PRAGMA table_info(documents)')
        .all()
        .map((column) => column.name),
    );

    // Step 4: Add each v3 document column if it does not already exist.
    for (const column of V3_DOCUMENT_COLUMNS) {
      if (existingDocumentColumns.has(column.name)) {
        skippedDocuments.push(column.name);
        continue;
      }
      const defaultClause =
        column.default === null ? '' : ` DEFAULT ${column.default}`;
      database.exec(
        `ALTER TABLE documents ADD COLUMN ${column.name} ${column.type}${defaultClause}`,
      );
      migratedDocuments.push(column.name);
    }

    // Step 5: Create v3 indexes if they do not already exist.
    const allExistingIndexes = new Set(
      database
        .prepare("SELECT name FROM sqlite_master WHERE type='index'")
        .all()
        .map((row) => row.name),
    );

    for (const indexDef of V3_INDEXES) {
      if (!allExistingIndexes.has(indexDef.name)) {
        database.exec(indexDef.sql);
        indexesCreated.push(indexDef.name);
      }
    }
  } finally {
    database.close();
  }

  return {
    migrated_chunks: migratedChunks,
    skipped_chunks: skippedChunks,
    migrated_documents: migratedDocuments,
    skipped_documents: skippedDocuments,
    indexes_created: indexesCreated,
  };
}

async function main() {
  const args = parseCliArgs(process.argv.slice(2));
  if (args.help) {
    printHelp({
      title: 'Schema migration: v1 → v2, v2 → v3',
      usage:
        'node scripts/semantic-index/migrate-schema.mjs [--json] [--database path]',
      options: [
        '--json           Emit JSON result on success.',
        '--database <path> Override database path.',
        '--help           Show this help.',
      ],
    });
    return;
  }

  try {
    const v2Result = migrateSchemaV1ToV2({ databasePath: args.database });
    const v3Result = migrateSchemaV2ToV3({ databasePath: args.database });
    writeJsonOrText(
      { v2: v2Result, v3: v3Result },
      Boolean(args.json),
      (payload) => {
        const parts = [];
        const v2 = payload.v2;
        const v3 = payload.v3;

        if (v2.migrated.length > 0)
          parts.push(`v2 migrated: ${v2.migrated.join(', ')}`);
        if (v2.skipped.length > 0)
          parts.push(`v2 skipped: ${v2.skipped.join(', ')}`);
        if (v2.indexes_created.length > 0)
          parts.push(`v2 indexes: ${v2.indexes_created.join(', ')}`);

        if (v3.migrated_chunks.length > 0)
          parts.push(`v3 chunks migrated: ${v3.migrated_chunks.join(', ')}`);
        if (v3.skipped_chunks.length > 0)
          parts.push(`v3 chunks skipped: ${v3.skipped_chunks.join(', ')}`);
        if (v3.migrated_documents.length > 0)
          parts.push(`v3 docs migrated: ${v3.migrated_documents.join(', ')}`);
        if (v3.skipped_documents.length > 0)
          parts.push(`v3 docs skipped: ${v3.skipped_documents.join(', ')}`);
        if (v3.indexes_created.length > 0)
          parts.push(`v3 indexes: ${v3.indexes_created.join(', ')}`);

        return parts.length > 0
          ? `Migration complete: ${parts.join('; ')}`
          : 'No migration needed — all columns and indexes already present.';
      },
    );
  } catch (error) {
    fail(
      error instanceof Error ? error.message : String(error),
      Boolean(args.json),
    );
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href)
  await main();
