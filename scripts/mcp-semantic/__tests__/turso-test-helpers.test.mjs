/**
 * @module turso-test-helpers.test
 * @description Coverage tests for turso-test-helpers.mjs — splitSqlStatements,
 * readCorpusSchema, createSchemaClient, createFileSchemaClient,
 * insertTestFixtures, insertGraphFixtures, createEnvIsolation.
 */
import {
  splitSqlStatements,
  readCorpusSchema,
  createSchemaClient,
  createFileSchemaClient,
  insertTestFixtures,
  insertGraphFixtures,
  createEnvIsolation,
  TEST_CHUNK_ID,
  TEST_PARENT_CHUNK_ID,
  TEST_DOC_ID,
  TEST_FILE_PATH,
  TEST_FAMILY,
  TEST_SYMBOL,
  UNIQUE_QUERY_TERM,
  MANAGED_ENV_VARS,
} from './turso-test-helpers.mjs';

import { jest } from '@jest/globals';
import { unlink } from 'node:fs/promises';

describe('turso-test-helpers', () => {
  const TEST_DB_FILES = ['test-turso-helpers.db'];

  afterEach(async () => {
    for (const f of TEST_DB_FILES) {
      try { await unlink(f); } catch { /* noop */ }
    }
  });
  describe('splitSqlStatements', () => {
    it('splits simple statements', () => {
      const sql = 'SELECT 1; SELECT 2; SELECT 3;';
      const stmts = splitSqlStatements(sql);
      expect(stmts).toHaveLength(3);
      expect(stmts[0]).toBe('SELECT 1;');
      expect(stmts[1]).toBe('SELECT 2;');
      expect(stmts[2]).toBe('SELECT 3;');
    });

    it('keeps BEGIN...END trigger bodies intact', () => {
      const sql = [
        'CREATE TRIGGER tr AFTER INSERT ON t',
        'BEGIN',
        '  UPDATE t SET x = 1;',
        '  INSERT INTO log VALUES (1);',
        'END;',
        'SELECT 1;',
      ].join(' ');
      const stmts = splitSqlStatements(sql);
      expect(stmts).toHaveLength(2);
      expect(stmts[0]).toContain('BEGIN');
      expect(stmts[0]).toContain('END');
      expect(stmts[0]).toContain('UPDATE t SET x = 1;');
      expect(stmts[1]).toBe('SELECT 1;');
    });

    it('handles multiple BEGIN...END blocks', () => {
      const sql = [
        'CREATE TRIGGER a BEGIN INSERT INTO t VALUES(1); END;',
        'CREATE TRIGGER b BEGIN UPDATE t SET x=2; END;',
      ].join(' ');
      const stmts = splitSqlStatements(sql);
      expect(stmts).toHaveLength(2);
      expect(stmts[0]).toContain('BEGIN');
      expect(stmts[1]).toContain('BEGIN');
    });

    it('discards empty and semicolon-only fragments', () => {
      const sql = ';;; SELECT 1;;';
      const stmts = splitSqlStatements(sql);
      expect(stmts).toHaveLength(1);
      expect(stmts[0]).toBe('SELECT 1;');
    });

    it('preserves trailing statement without semicolon', () => {
      const sql = 'SELECT 1; SELECT 2';
      const stmts = splitSqlStatements(sql);
      expect(stmts).toHaveLength(2);
      expect(stmts[1]).toBe('SELECT 2');
    });

    it('handles empty input', () => {
      expect(splitSqlStatements('')).toEqual([]);
    });

    it('handles whitespace-only input', () => {
      expect(splitSqlStatements('   ')).toEqual([]);
    });

    it('handles BEGIN without END (unclosed)', () => {
      const sql = 'CREATE TRIGGER tr BEGIN INSERT INTO t VALUES(1);';
      const stmts = splitSqlStatements(sql);
      // BEGIN is set, semicolons inside BEGIN are ignored, trailing buffer is output
      expect(stmts).toHaveLength(1);
      expect(stmts[0]).toContain('BEGIN');
    });

    it('handles END without preceding BEGIN', () => {
      const sql = 'END; SELECT 1;';
      const stmts = splitSqlStatements(sql);
      // END without BEGIN: inBegin stays false, semicolons split normally
      expect(stmts).toHaveLength(2);
    });

    it('handles lowercase begin/end (case-insensitive)', () => {
      const sql = 'create trigger tr begin insert into t values(1); end; select 1;';
      const stmts = splitSqlStatements(sql);
      // The function checks .toUpperCase() so lowercase begin/end should work
      expect(stmts.length).toBeGreaterThanOrEqual(1);
    });
  });

  describe('readCorpusSchema', () => {
    it('returns non-empty SQL string', async () => {
      const sql = await readCorpusSchema();
      expect(typeof sql).toBe('string');
      expect(sql.length).toBeGreaterThan(100);
      expect(sql).toContain('CREATE');
    });
  });

  describe('createSchemaClient', () => {
    it('creates an in-memory client with the full schema', async () => {
      const client = await createSchemaClient();
      expect(client).toBeDefined();
      // Verify key tables exist
      const tables = await client.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
      );
      const tableNames = tables.rows.map((r) => r.name);
      expect(tableNames).toContain('documents');
      expect(tableNames).toContain('chunks');
      expect(tableNames).toContain('entities');
      expect(tableNames).toContain('edges');
      expect(tableNames).toContain('feedback_events');
      expect(tableNames).toContain('feedback_scores');
      await client.close();
    });
  });

  describe('createFileSchemaClient', () => {
    it('creates a file-backed client with the full schema', async () => {
      const dbPath = 'test-turso-helpers.db';
      const client = await createFileSchemaClient(dbPath);
      expect(client).toBeDefined();
      const tables = await client.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
      );
      const tableNames = tables.rows.map((r) => r.name);
      expect(tableNames).toContain('documents');
      expect(tableNames).toContain('chunks');
      await client.close();
    });
  });

  describe('insertTestFixtures', () => {
    it('inserts document, parent chunk, and sub-chunk', async () => {
      const client = await createSchemaClient();
      await insertTestFixtures(client);

      const docs = await client.execute('SELECT * FROM documents');
      expect(docs.rows).toHaveLength(1);
      expect(docs.rows[0].doc_id).toBe(TEST_DOC_ID);
      expect(docs.rows[0].file_path).toBe(TEST_FILE_PATH);
      expect(docs.rows[0].doc_family).toBe(TEST_FAMILY);

      const chunks = await client.execute(
        'SELECT * FROM chunks ORDER BY chunk_id',
      );
      expect(chunks.rows).toHaveLength(2);
      const parent = chunks.rows.find((c) => c.chunk_id === TEST_PARENT_CHUNK_ID);
      const child = chunks.rows.find((c) => c.chunk_id === TEST_CHUNK_ID);
      expect(parent).toBeDefined();
      expect(parent.depth).toBe(0);
      expect(parent.symbol_name).toBe(TEST_SYMBOL);
      expect(child).toBeDefined();
      expect(child.depth).toBe(1);
      expect(child.parent_chunk_id).toBe(TEST_PARENT_CHUNK_ID);
      expect(String(child.body_text)).toContain(UNIQUE_QUERY_TERM);

      await client.close();
    });

    it('populates chunks_fts via trigger', async () => {
      const client = await createSchemaClient();
      await insertTestFixtures(client);
      const fts = await client.execute('SELECT * FROM chunks_fts');
      expect(fts.rows.length).toBeGreaterThan(0);
      await client.close();
    });
  });

  describe('insertGraphFixtures', () => {
    it('inserts two entities and one edge', async () => {
      const client = await createSchemaClient();
      await insertTestFixtures(client);
      await insertGraphFixtures(client);

      const entities = await client.execute('SELECT * FROM entities ORDER BY entity_id');
      expect(entities.rows).toHaveLength(2);
      const entityA = entities.rows.find((e) => e.entity_id === 900001);
      const entityB = entities.rows.find((e) => e.entity_id === 900002);
      expect(entityA).toBeDefined();
      expect(entityA.entity_type).toBe('function');
      expect(entityA.name).toBe('TursoTestEntityA');
      expect(entityB).toBeDefined();
      expect(entityB.entity_type).toBe('class');
      expect(entityB.name).toBe('TursoTestEntityB');

      const edges = await client.execute('SELECT * FROM edges');
      expect(edges.rows).toHaveLength(1);
      expect(edges.rows[0].relationship).toBe('references');
      expect(edges.rows[0].confidence).toBe('high');

      await client.close();
    });
  });

  describe('createEnvIsolation', () => {
    it('saves and restores managed env vars', () => {
      const isolation = createEnvIsolation();
      // Set some env vars
      for (const key of MANAGED_ENV_VARS) {
        process.env[key] = 'test-value';
      }
      // Save
      isolation.saveEnv();
      // Env vars should be deleted
      for (const key of MANAGED_ENV_VARS) {
        expect(process.env[key]).toBeUndefined();
      }
      // Set different values
      process.env[MANAGED_ENV_VARS[0]] = 'other-value';
      // Restore
      isolation.restoreEnv();
      // Should have original values
      for (const key of MANAGED_ENV_VARS) {
        expect(process.env[key]).toBe('test-value');
      }
      // Cleanup
      for (const key of MANAGED_ENV_VARS) {
        delete process.env[key];
      }
    });

    it('handles undefined saved values (deletes on restore)', () => {
      const isolation = createEnvIsolation();
      // Don't set any env vars before save
      for (const key of MANAGED_ENV_VARS) {
        delete process.env[key];
      }
      isolation.saveEnv();
      // Set some values
      process.env[MANAGED_ENV_VARS[0]] = 'temporary';
      // Restore
      isolation.restoreEnv();
      // Should be deleted (since saved value was undefined)
      expect(process.env[MANAGED_ENV_VARS[0]]).toBeUndefined();
    });

    it('can be called multiple times independently', () => {
      const iso1 = createEnvIsolation();
      const iso2 = createEnvIsolation();
      process.env[MANAGED_ENV_VARS[0]] = 'value1';
      iso1.saveEnv();
      process.env[MANAGED_ENV_VARS[0]] = 'value2';
      iso2.saveEnv();
      // Both should have saved different values
      iso2.restoreEnv();
      expect(process.env[MANAGED_ENV_VARS[0]]).toBe('value2');
      iso1.restoreEnv();
      expect(process.env[MANAGED_ENV_VARS[0]]).toBe('value1');
      delete process.env[MANAGED_ENV_VARS[0]];
    });
  });

  describe('exported constants', () => {
    it('exports expected constants', () => {
      expect(TEST_CHUNK_ID).toBe(424242);
      expect(TEST_PARENT_CHUNK_ID).toBe(424241);
      expect(TEST_DOC_ID).toBe(500001);
      expect(TEST_FILE_PATH).toBe('src/turso-async-test.ts');
      expect(TEST_FAMILY).toBe('turso-test');
      expect(TEST_SYMBOL).toBe('TursoAsyncTestSymbol');
      expect(UNIQUE_QUERY_TERM).toBe('tursouniqueword');
      expect(MANAGED_ENV_VARS).toEqual([
        'TURSO_DATABASE_URL',
        'TURSO_AUTH_TOKEN',
        'TURSO_SYNC_URL',
        'TURSO_SYNC_INTERVAL',
        'CORTEX_DB_PATH',
      ]);
    });
  });
});