/**
 * @module list-families.test
 * @description Coverage tests for list-families.mjs — branch coverage for
 * options.client vs getTursoClient paths.
 */
import {
  createSchemaClient,
  insertTestFixtures,
  createEnvIsolation,
  TEST_FAMILY,
} from '../__tests__/turso-test-helpers.mjs';

const { listFamilies } = await import('./list-families.mjs');

describe('list-families', () => {
  const { saveEnv, restoreEnv } = createEnvIsolation();
  let client;

  beforeEach(async () => {
    saveEnv();
    client = await createSchemaClient();
    await insertTestFixtures(client);
  });

  afterEach(async () => {
    restoreEnv();
    if (client) {
      try {
        await client.close();
      } catch {
        /* noop */
      }
    }
  });

  it('returns families when using injected client', async () => {
    const result = await listFamilies({ client });

    expect(result.families).toHaveLength(1);
    expect(result.families[0].family).toBe(TEST_FAMILY);
    expect(result.families[0].documents).toBe(1);
    expect(result.families[0].chunks).toBe(2);
  });

  it('returns empty families when no documents exist', async () => {
    const emptyClient = await createSchemaClient();
    try {
      const result = await listFamilies({ client: emptyClient });
      expect(result.families).toEqual([]);
    } finally {
      await emptyClient.close();
    }
  });

  it('returns multiple families sorted alphabetically', async () => {
    // Add a second family
    await client.execute({
      sql: `INSERT INTO documents (doc_id, file_path, doc_family, mtime_ms, file_size, sha256, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)`,
      args: [500002, 'src/another.ts', 'aaa-family', 2000, 300, 'sha2', 2000],
    });
    await client.execute({
      sql: `INSERT INTO chunks (chunk_id, doc_id, chunk_index, heading_path, body_text, char_start, char_end, parent_chunk_id, depth)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      args: [424243, 500002, 0, 'Another', 'another body', 0, 10, null, 0],
    });

    const result = await listFamilies({ client });

    expect(result.families).toHaveLength(2);
    // Sorted alphabetically: 'aaa-family' before 'turso-test'
    expect(result.families[0].family).toBe('aaa-family');
    expect(result.families[1].family).toBe(TEST_FAMILY);
  });

  it('counts chunks correctly with LEFT JOIN (family with no chunks)', async () => {
    // Add a document with no chunks
    await client.execute({
      sql: `INSERT INTO documents (doc_id, file_path, doc_family, mtime_ms, file_size, sha256, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)`,
      args: [
        500003,
        'src/no-chunks.ts',
        'no-chunks-family',
        3000,
        100,
        'sha3',
        3000,
      ],
    });

    const result = await listFamilies({ client });

    const noChunksFamily = result.families.find(
      (f) => f.family === 'no-chunks-family',
    );
    expect(noChunksFamily).toBeDefined();
    expect(noChunksFamily.documents).toBe(1);
    expect(noChunksFamily.chunks).toBe(0);
  });

  it('uses getTursoClient when no client provided', async () => {
    // This exercises the `await getTursoClient(options.databasePath)` branch.
    // We pass databasePath to a :memory: client via setTursoClient.
    const { setTursoClient, closeTursoClient } =
      await import('./cortex-db.mjs');
    const dbPath = ':memory:test-list-families';
    setTursoClient(dbPath, client);
    try {
      const result = await listFamilies({ databasePath: dbPath });
      expect(result.families).toHaveLength(1);
    } finally {
      closeTursoClient(dbPath);
    }
  });
});
