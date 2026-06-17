/**
 * @module ann.test
 * @description Red tests for ANN index determinism contracts.
 *
 * The Repo Cortex ANN index must return the same nearest-neighbour ordering
 * when the same embedding is queried repeatedly against the same index. These
 * tests expose non-deterministic tie-breaking or seeding before the
 * implementation pins ordering to a deterministic function of the index.
 *
 * The test executes the ESM implementation modules in a child Node process so
 * that the ts-jest project for this folder does not need to transform .mjs
 * source files.
 */
import { execFileSync } from 'node:child_process';
import os from 'node:os';
import path from 'node:path';
import fs from 'node:fs';
import Database from 'better-sqlite3';

const REPO_ROOT = path.resolve();

interface AnnDeterminismResult {
  firstIds: number[];
  secondIds: number[];
}

/**
 * Evaluate a short ESM snippet in a child Node process rooted at the repo root.
 */
const runModuleEvaluation = <Result>(source: string): Result => {
  const output = execFileSync(
    process.execPath,
    ['--input-type=module', '--eval', source],
    {
      cwd: REPO_ROOT,
      encoding: 'utf8',
    },
  );
  const trimmed = output.trim();
  if (trimmed.length === 0) {
    throw new Error('Module evaluation produced empty output');
  }
  return JSON.parse(trimmed) as Result;
};

describe('ann-index determinism', () => {
  it('returns identical HNSW result order across repeated queries with the same embedding', () => {
    const tempDir = fs.mkdtempSync(
      path.join(os.tmpdir(), 'ann-determinism-red-'),
    );
    const embeddingsDatabasePath = path.join(tempDir, 'embeddings.sqlite');
    const indexFilePath = path.join(tempDir, 'hnsw.index');

    const db = new Database(embeddingsDatabasePath);
    try {
      db.exec(`
        CREATE TABLE chunk_embeddings (
          chunk_id INTEGER NOT NULL,
          model_id TEXT NOT NULL,
          dimension INTEGER NOT NULL,
          embedding BLOB NOT NULL,
          chunk_sha256 TEXT,
          created_at TEXT NOT NULL DEFAULT(datetime('now')),
          updated_at TEXT NOT NULL DEFAULT(datetime('now')),
          PRIMARY KEY(chunk_id, model_id)
        );
      `);

      const insert = db.prepare(
        'INSERT INTO chunk_embeddings (chunk_id, model_id, dimension, embedding) VALUES (?, ?, ?, ?)',
      );
      for (let chunkId = 10; chunkId <= 12; chunkId++) {
        const vector = new Float32Array([chunkId, chunkId + 1]);
        insert.run(
          chunkId,
          'mock',
          2,
          Buffer.from(vector.buffer, vector.byteOffset, vector.byteLength),
        );
      }
    } finally {
      db.close();
    }

    try {
      const result = runModuleEvaluation<AnnDeterminismResult>(`
        import {
          __hnswTestSeam,
          __refreshHnswAvailability,
        } from './scripts/mcp-semantic/tools/ann-strategy.mjs';
        import {
          buildAnnIndex,
          queryHnswIndex,
        } from './scripts/mcp-semantic/tools/ann-index.mjs';
        import fs from 'node:fs';

        let queryCount = 0;
        class MockHierarchicalNSW {
          constructor() {}
          initIndex() {}
          addPoint() {}
          setEf() {}
          writeIndexSync() {}
          readIndexSync() {}
          searchKnn() {
            queryCount += 1;
            // Deliberately return a different permutation on each call so
            // that any non-deterministic ordering in the caller is visible.
            if (queryCount % 2 === 0) {
              return { neighbors: [1, 2, 0], distances: [0.3, 0.1, 0.2] };
            }
            return { neighbors: [2, 0, 1], distances: [0.1, 0.2, 0.3] };
          }
        }

        __hnswTestSeam.importFn = async () => ({
          HierarchicalNSW: MockHierarchicalNSW,
        });
        await __refreshHnswAvailability();

        await buildAnnIndex({
          embeddingsDatabasePath: ${JSON.stringify(embeddingsDatabasePath)},
          indexFilePath: ${JSON.stringify(indexFilePath)},
          modelId: 'mock',
          dimension: 2,
          hnswAvailable: true,
          forceStrategy: 'hnsw',
        });

        const first = await queryHnswIndex(new Float32Array([1, 0]), {
          embeddingsDatabasePath: ${JSON.stringify(embeddingsDatabasePath)},
          k: 3,
          modelId: 'mock',
          dimension: 2,
        });

        const second = await queryHnswIndex(new Float32Array([1, 0]), {
          embeddingsDatabasePath: ${JSON.stringify(embeddingsDatabasePath)},
          k: 3,
          modelId: 'mock',
          dimension: 2,
        });

        fs.rmSync(${JSON.stringify(tempDir)}, { recursive: true, force: true });

        console.log(JSON.stringify({
          firstIds: first.map((r) => r.chunk_id),
          secondIds: second.map((r) => r.chunk_id),
        }));
      `);

      expect(result.firstIds).toEqual(result.secondIds);
    } finally {
      try {
        fs.rmSync(tempDir, { recursive: true, force: true });
      } catch {
        // Best-effort cleanup; ignore failures on Windows locked handles.
      }
    }
  });
});
