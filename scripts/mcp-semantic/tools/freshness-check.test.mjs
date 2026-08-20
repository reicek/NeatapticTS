/**
 * @module freshness-check.test
 * @description Coverage tests for freshness-check.mjs tool.
 */
import { jest } from '@jest/globals';
import { createClient } from '@libsql/client';

// Mock the upstream freshness module before importing the tool.
const mockGetFreshnessProof = jest.fn();
const mockIsFreshDocument = jest.fn();

jest.unstable_mockModule('../../../rag-index/freshness.mjs', () => ({
  getFreshnessProof: mockGetFreshnessProof,
  isFreshDocument: mockIsFreshDocument,
  __esModule: true,
}));

jest.unstable_mockModule('../../../rag-index/init-schema.mjs', () => ({
  repoRoot: '/test-repo',
  defaultDatabasePath: '/test/cortex.db',
  __esModule: true,
}));

const { freshnessCheck } = await import('./freshness-check.mjs');

/**
 * Creates an in-memory client with the documents schema for freshness checks.
 * @param {object[]} docs - Documents to insert.
 * @returns {Promise<import('@libsql/client').Client>}
 */
async function createFreshnessClient(docs = []) {
  const client = createClient({ url: ':memory:' });
  await client.execute(`
    CREATE TABLE documents (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      file_path TEXT NOT NULL UNIQUE,
      mtime_ms INTEGER,
      file_size INTEGER,
      sha256 TEXT,
      indexed_at TEXT
    )
  `);
  for (const doc of docs) {
    await client.execute({
      sql: 'INSERT INTO documents (file_path, mtime_ms, file_size, sha256, indexed_at) VALUES (?, ?, ?, ?, ?)',
      args: [
        doc.file_path,
        doc.mtime_ms ?? null,
        doc.file_size ?? null,
        doc.sha256 ?? null,
        doc.indexed_at ?? null,
      ],
    });
  }
  return client;
}

describe('freshness-check', () => {
  beforeEach(() => {
    mockGetFreshnessProof.mockReset();
    mockIsFreshDocument.mockReset();
  });

  describe('freshnessCheck — single file_path', () => {
    it('throws when file_path is not found in index', async () => {
      const client = await createFreshnessClient([]);
      try {
        await expect(
          freshnessCheck({ file_path: 'missing.ts', client }),
        ).rejects.toThrow('Document not found: missing.ts');
      } finally {
        await client.close();
      }
    });

    it('returns fresh result for a found document', async () => {
      const client = await createFreshnessClient([
        {
          file_path: 'src/index.ts',
          mtime_ms: 1000,
          file_size: 500,
          sha256: 'abc123',
          indexed_at: '2024-01-01T00:00:00.000Z',
        },
      ]);
      mockGetFreshnessProof.mockResolvedValue({
        mtime_ms: 1000,
        file_size: 500,
        sha256: 'abc123',
      });
      mockIsFreshDocument.mockReturnValue(true);
      try {
        const result = await freshnessCheck({
          file_path: 'src/index.ts',
          client,
        });

        expect(result.fresh).toBe(true);
        expect(result.stale).toEqual([]);
        expect(result.documents).toHaveLength(1);
        expect(result.documents[0].file_path).toBe('src/index.ts');
        expect(result.documents[0].fresh).toBe(true);
        expect(result.documents[0].indexed).toEqual({
          mtime_ms: 1000,
          file_size: 500,
          sha256: 'abc123',
        });
        expect(result.documents[0].current).toEqual({
          mtime_ms: 1000,
          file_size: 500,
          sha256: 'abc123',
        });
        expect(result.documents[0].freshness).toEqual({
          timestamp: expect.any(Number),
          stale: false,
          last_update_source: 'filesystem',
          last_sync: null,
          sync_lag_ms: null,
        });
      } finally {
        await client.close();
      }
    });

    it('returns stale result when isFreshDocument returns false', async () => {
      const client = await createFreshnessClient([
        {
          file_path: 'src/stale.ts',
          mtime_ms: 2000,
          file_size: 500,
          sha256: 'abc123',
          indexed_at: '2024-01-01T00:00:00.000Z',
        },
      ]);
      mockGetFreshnessProof.mockResolvedValue({
        mtime_ms: 5000,
        file_size: 600,
        sha256: 'def456',
      });
      mockIsFreshDocument.mockReturnValue(false);
      try {
        const result = await freshnessCheck({
          file_path: 'src/stale.ts',
          client,
        });

        expect(result.fresh).toBe(false);
        expect(result.stale).toEqual(['src/stale.ts']);
        expect(result.freshness.stale).toBe(true);
        expect(result.documents[0].fresh).toBe(false);
        expect(result.documents[0].freshness.stale).toBe(true);
      } finally {
        await client.close();
      }
    });

    it('uses supplied freshnessProof instead of calling getFreshnessProof', async () => {
      const client = await createFreshnessClient([
        {
          file_path: 'src/index.ts',
          mtime_ms: 1000,
          file_size: 500,
          sha256: 'abc123',
          indexed_at: '2024-01-01T00:00:00.000Z',
        },
      ]);
      const suppliedProof = {
        mtime_ms: 1000,
        file_size: 500,
        sha256: 'abc123',
      };
      mockIsFreshDocument.mockReturnValue(true);
      try {
        await freshnessCheck({
          file_path: 'src/index.ts',
          client,
          freshnessProof: suppliedProof,
        });

        expect(mockGetFreshnessProof).not.toHaveBeenCalled();
        expect(mockIsFreshDocument).toHaveBeenCalledWith(
          expect.objectContaining({ file_path: 'src/index.ts' }),
          suppliedProof,
        );
      } finally {
        await client.close();
      }
    });
  });

  describe('freshnessCheck — all documents (no file_path)', () => {
    it('checks all documents and returns summary', async () => {
      const client = await createFreshnessClient([
        {
          file_path: 'src/fresh.ts',
          mtime_ms: 1000,
          file_size: 500,
          sha256: 'aaa',
          indexed_at: '2024-01-01T00:00:00.000Z',
        },
        {
          file_path: 'src/stale.ts',
          mtime_ms: 2000,
          file_size: 600,
          sha256: 'bbb',
          indexed_at: '2024-01-01T00:00:00.000Z',
        },
      ]);
      mockGetFreshnessProof.mockResolvedValue({
        mtime_ms: 1000,
        file_size: 500,
        sha256: 'aaa',
      });
      mockIsFreshDocument.mockReturnValueOnce(true).mockReturnValueOnce(false);
      try {
        const result = await freshnessCheck({ client });

        expect(result.fresh).toBe(false);
        expect(result.stale).toEqual(['src/stale.ts']);
        expect(result.documents).toHaveLength(2);
        expect(result.freshness.stale).toBe(true);
      } finally {
        await client.close();
      }
    });

    it('returns fresh=true when all documents are fresh', async () => {
      const client = await createFreshnessClient([
        {
          file_path: 'src/a.ts',
          mtime_ms: 1000,
          file_size: 500,
          sha256: 'aaa',
          indexed_at: '2024-01-01T00:00:00.000Z',
        },
      ]);
      mockGetFreshnessProof.mockResolvedValue({
        mtime_ms: 1000,
        file_size: 500,
        sha256: 'aaa',
      });
      mockIsFreshDocument.mockReturnValue(true);
      try {
        const result = await freshnessCheck({ client });

        expect(result.fresh).toBe(true);
        expect(result.stale).toEqual([]);
        expect(result.freshness.stale).toBe(false);
      } finally {
        await client.close();
      }
    });

    it('returns fresh=true when no documents exist', async () => {
      const client = await createFreshnessClient([]);
      try {
        const result = await freshnessCheck({ client });

        expect(result.fresh).toBe(true);
        expect(result.stale).toEqual([]);
        expect(result.documents).toEqual([]);
      } finally {
        await client.close();
      }
    });
  });

  describe('freshnessCheck — file_path edge cases', () => {
    it('treats file_path="." as no file_path (all documents)', async () => {
      const client = await createFreshnessClient([
        {
          file_path: 'src/a.ts',
          mtime_ms: 1000,
          file_size: 500,
          sha256: 'aaa',
          indexed_at: '2024-01-01T00:00:00.000Z',
        },
      ]);
      mockGetFreshnessProof.mockResolvedValue({});
      mockIsFreshDocument.mockReturnValue(true);
      try {
        const result = await freshnessCheck({ file_path: '.', client });

        expect(result.documents).toHaveLength(1);
      } finally {
        await client.close();
      }
    });

    it('treats file_path="" as no file_path (all documents)', async () => {
      const client = await createFreshnessClient([
        {
          file_path: 'src/a.ts',
          mtime_ms: 1000,
          file_size: 500,
          sha256: 'aaa',
          indexed_at: '2024-01-01T00:00:00.000Z',
        },
      ]);
      mockGetFreshnessProof.mockResolvedValue({});
      mockIsFreshDocument.mockReturnValue(true);
      try {
        const result = await freshnessCheck({ file_path: '', client });

        expect(result.documents).toHaveLength(1);
      } finally {
        await client.close();
      }
    });

    it('treats file_path="  " as no file_path (all documents)', async () => {
      const client = await createFreshnessClient([
        {
          file_path: 'src/a.ts',
          mtime_ms: 1000,
          file_size: 500,
          sha256: 'aaa',
          indexed_at: '2024-01-01T00:00:00.000Z',
        },
      ]);
      mockGetFreshnessProof.mockResolvedValue({});
      mockIsFreshDocument.mockReturnValue(true);
      try {
        const result = await freshnessCheck({ file_path: '  ', client });

        expect(result.documents).toHaveLength(1);
      } finally {
        await client.close();
      }
    });

    it('treats undefined file_path as all documents', async () => {
      const client = await createFreshnessClient([]);
      try {
        const result = await freshnessCheck({ client });

        expect(result.documents).toEqual([]);
      } finally {
        await client.close();
      }
    });
  });
});
