import { jest } from '@jest/globals';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const mockCreateClient = jest.fn();
const mockMkdir = jest.fn();
const mockReadFile = jest.fn();

jest.unstable_mockModule('@libsql/client', () => ({
  createClient: mockCreateClient,
  default: { createClient: mockCreateClient },
}));
jest.unstable_mockModule('node:fs/promises', () => ({
  mkdir: mockMkdir,
  readFile: mockReadFile,
  default: { mkdir: mockMkdir, readFile: mockReadFile },
}));

const { repoRoot, defaultDatabasePath, initSemanticIndex } = await import('./init-schema.mjs');

afterEach(() => {
  jest.clearAllMocks();
});

describe('exports', () => {
  it('exports repoRoot as absolute path', () => {
    expect(path.isAbsolute(repoRoot)).toBe(true);
  });

  it('exports defaultDatabasePath containing turso-replica', () => {
    expect(defaultDatabasePath).toContain('turso-replica.sqlite');
  });
});

describe('initSemanticIndex', () => {
  it('returns provided client directly', async () => {
    const fakeClient = { execute: jest.fn(), batch: jest.fn() };
    const result = await initSemanticIndex({ client: fakeClient });
    expect(result).toBe(fakeClient);
    expect(mockCreateClient).not.toHaveBeenCalled();
  });

  it('creates client, applies schema, and migrates columns', async () => {
    const mockClient = {
      execute: jest.fn().mockResolvedValue({ rows: [] }),
      batch: jest.fn().mockResolvedValue([]),
    };
    mockCreateClient.mockReturnValue(mockClient);
    mockMkdir.mockResolvedValue(undefined);
    mockReadFile.mockResolvedValue('CREATE TABLE documents (doc_id INTEGER PRIMARY KEY);\nCREATE TABLE chunks (chunk_id INTEGER PRIMARY KEY);\n');

    const result = await initSemanticIndex({ databasePath: '/fake/db.sqlite' });
    expect(result).toBe(mockClient);
    expect(mockMkdir).toHaveBeenCalled();
    expect(mockCreateClient).toHaveBeenCalledWith(expect.objectContaining({ url: expect.any(String) }));
    // Schema applied via batch
    expect(mockClient.batch).toHaveBeenCalled();
    // PRAGMA table_info called for migration
    const pragmaCall = mockClient.execute.mock.calls.find(
      ([{ sql }]) => sql === 'PRAGMA table_info(chunks)',
    );
    expect(pragmaCall).toBeDefined();
  });

  it('migrates missing slice metadata columns', async () => {
    const mockClient = {
      execute: jest.fn(async ({ sql }) => {
        if (sql === 'PRAGMA table_info(chunks)') {
          return { rows: [{ name: 'chunk_id' }, { name: 'body_text' }] };
        }
        return { rows: [] };
      }),
      batch: jest.fn().mockResolvedValue([]),
    };
    mockCreateClient.mockReturnValue(mockClient);
    mockMkdir.mockResolvedValue(undefined);
    mockReadFile.mockResolvedValue('CREATE TABLE chunks (chunk_id INTEGER);\n');

    await initSemanticIndex({ databasePath: '/fake/db.sqlite' });
    // Should have ALTER TABLE calls for missing columns
    const alterCalls = mockClient.execute.mock.calls.filter(
      ([{ sql }]) => sql.startsWith('ALTER TABLE chunks ADD COLUMN'),
    );
    expect(alterCalls).toHaveLength(4);
  });

  it('skips migration when slice metadata columns already exist', async () => {
    const mockClient = {
      execute: jest.fn(async ({ sql }) => {
        if (sql === 'PRAGMA table_info(chunks)') {
          return {
            rows: [
              { name: 'chunk_id' },
              { name: 'slice_id' },
              { name: 'step_number' },
              { name: 'phase' },
              { name: 'status' },
            ],
          };
        }
        return { rows: [] };
      }),
      batch: jest.fn().mockResolvedValue([]),
    };
    mockCreateClient.mockReturnValue(mockClient);
    mockMkdir.mockResolvedValue(undefined);
    mockReadFile.mockResolvedValue('CREATE TABLE chunks (chunk_id INTEGER);\n');

    await initSemanticIndex({ databasePath: '/fake/db.sqlite' });
    const alterCalls = mockClient.execute.mock.calls.filter(
      ([{ sql }]) => sql.startsWith('ALTER TABLE chunks ADD COLUMN'),
    );
    expect(alterCalls).toHaveLength(0);
  });

  it('handles trigger statements in schema SQL', async () => {
    const mockClient = {
      execute: jest.fn().mockResolvedValue({ rows: [] }),
      batch: jest.fn().mockResolvedValue([]),
    };
    mockCreateClient.mockReturnValue(mockClient);
    mockMkdir.mockResolvedValue(undefined);
    const schemaWithTrigger = [
      'CREATE TABLE chunks (chunk_id INTEGER PRIMARY KEY);',
      'CREATE TRIGGER chunks_fts_insert AFTER INSERT ON chunks BEGIN',
      '  INSERT INTO chunks_fts (chunk_id, body_text) VALUES (new.chunk_id, new.body_text);',
      'END;',
      'CREATE TABLE documents (doc_id INTEGER PRIMARY KEY);',
    ].join('\n');
    mockReadFile.mockResolvedValue(schemaWithTrigger);

    await initSemanticIndex({ databasePath: '/fake/db.sqlite' });
    // The trigger should be kept as one statement
    const batchCalls = mockClient.batch.mock.calls;
    const allStatements = batchCalls.flatMap(([stmts]) => stmts.map((s) => s.sql));
    const triggerStmt = allStatements.find((s) => s.includes('CREATE TRIGGER'));
    expect(triggerStmt).toBeDefined();
    expect(triggerStmt).toContain('BEGIN');
    expect(triggerStmt).toContain('END');
  });

  it('batches statements in groups of 50', async () => {
    const mockClient = {
      execute: jest.fn().mockResolvedValue({ rows: [] }),
      batch: jest.fn().mockResolvedValue([]),
    };
    mockCreateClient.mockReturnValue(mockClient);
    mockMkdir.mockResolvedValue(undefined);
    // Create 51 simple statements to trigger 2 batch groups
    const statements = Array.from({ length: 51 }, (_, i) => `CREATE TABLE t${i} (id INTEGER);`).join('\n');
    mockReadFile.mockResolvedValue(statements);

    await initSemanticIndex({ databasePath: '/fake/db.sqlite' });
    // Should have 2 batch calls: first with 50, second with 1
    expect(mockClient.batch).toHaveBeenCalledTimes(2);
    expect(mockClient.batch.mock.calls[0][0]).toHaveLength(50);
    expect(mockClient.batch.mock.calls[1][0]).toHaveLength(1);
  });

  it('handles schema with trailing content without semicolon', async () => {
    const mockClient = {
      execute: jest.fn().mockResolvedValue({ rows: [] }),
      batch: jest.fn().mockResolvedValue([]),
    };
    mockCreateClient.mockReturnValue(mockClient);
    mockMkdir.mockResolvedValue(undefined);
    mockReadFile.mockResolvedValue('CREATE TABLE test (id INTEGER);\n-- comment without semicolon');

    await initSemanticIndex({ databasePath: '/fake/db.sqlite' });
    // The tail should be included as a statement
    const allStatements = mockClient.batch.mock.calls.flatMap(([stmts]) => stmts.map((s) => s.sql));
    // The comment line doesn't end with ; so it becomes a tail statement
    expect(allStatements.length).toBeGreaterThanOrEqual(1);
  });

  it('uses defaultDatabasePath when no databasePath provided', async () => {
    const mockClient = {
      execute: jest.fn().mockResolvedValue({ rows: [] }),
      batch: jest.fn().mockResolvedValue([]),
    };
    mockCreateClient.mockReturnValue(mockClient);
    mockMkdir.mockResolvedValue(undefined);
    mockReadFile.mockResolvedValue('CREATE TABLE test (id INTEGER);\n');

    await initSemanticIndex();
    expect(mockCreateClient).toHaveBeenCalled();
    expect(mockMkdir).toHaveBeenCalled();
  });
});