/**
 * @module schema-session.turso.test
 * @description Red tests for Turso async-migration of init-schema and
 *              migrate-schema scripts.
 *
 * These tests verify that the schema and session scripts accept an optional
 * `client` parameter (a `@libsql/client` Client) and use it for database
 * operations instead of creating a synchronous native SQLite Database.
 *
 * RED PHASE: the scripts currently use sync native SQLite and ignore the
 * `client` parameter. Every test fails because the function either:
 *  - creates its own native SQLite DB and returns that instead of the client, or
 *  - is synchronous instead of asynchronous, or
 *  - throws because it tries to open a non-existent DB path instead of using
 *    the injected client.
 *
 * Pure .mjs test — runs via Jest ESM project `semantic-index-mjs`.
 */

import {
  createSchemaClient,
  createEnvIsolation,
} from './turso-test-helpers.mjs';
import os from 'node:os';
import path from 'node:path';

const { saveEnv, restoreEnv } = createEnvIsolation();

// ---------------------------------------------------------------------------
// init-schema.mjs — initSemanticIndex
// ---------------------------------------------------------------------------

describe('initSemanticIndex (Turso async migration)', () => {
  beforeEach(() => saveEnv());
  afterEach(() => restoreEnv());

  it('should return the provided libSQL client instead of creating a native SQLite database', async () => {
    const { initSemanticIndex } = await import('../init-schema.mjs');
    const client = await createSchemaClient();
    const tmpPath = path.join(
      os.tmpdir(),
      `turso-test-init-${Date.now()}.sqlite`,
    );

    // After migration, the function should return the client (or use it
    // instead of creating a new native SQLite Database).
    // Currently the function ignores client, creates new Database(tmpPath),
    // and returns that native SQLite object → not the same as client → RED.
    const result = await initSemanticIndex({ client, databasePath: tmpPath });
    expect(result).toBe(client);
  });
});

// ---------------------------------------------------------------------------
// migrate-schema.mjs — migrateSchemaV1ToV2, migrateSchemaV2ToV3
// ---------------------------------------------------------------------------

describe('migrateSchemaV1ToV2 (Turso async migration)', () => {
  beforeEach(() => saveEnv());
  afterEach(() => restoreEnv());

  it('should be async and accept a libSQL client', async () => {
    const { migrateSchemaV1ToV2 } = await import('../migrate-schema.mjs');
    const client = await createSchemaClient();
    const tmpPath = path.join(
      os.tmpdir(),
      `turso-test-mig12-${Date.now()}.sqlite`,
    );

    // After migration, the function should be async and use the client.
    // Currently the function is sync (returns an object, not a Promise) and
    // tries to open tmpPath with the native driver → returns sync result → RED.
    const result = migrateSchemaV1ToV2({ client, databasePath: tmpPath });
    expect(result).toBeInstanceOf(Promise);
  });
});

describe('migrateSchemaV2ToV3 (Turso async migration)', () => {
  beforeEach(() => saveEnv());
  afterEach(() => restoreEnv());

  it('should be async and accept a libSQL client', async () => {
    const { migrateSchemaV2ToV3 } = await import('../migrate-schema.mjs');
    const client = await createSchemaClient();
    const tmpPath = path.join(
      os.tmpdir(),
      `turso-test-mig23-${Date.now()}.sqlite`,
    );

    // After migration, the function should be async and use the client.
    // Currently the function is sync (returns an object, not a Promise) and
    // tries to open tmpPath with the native driver → returns sync result → RED.
    const result = migrateSchemaV2ToV3({ client, databasePath: tmpPath });
    expect(result).toBeInstanceOf(Promise);
  });
});

