/**
 * @module freshness-check.test
 * @description Red tests for corpus freshness transparency.
 *
 * The freshness-check tool currently returns per-document indexed/current proof
 * data but does not surface a canonical freshness proof stanza that search
 * responses can embed. These tests capture the required contract: every
 * freshness report must carry a top-level `freshness` object (timestamp,
 * stale flag, last update source) and propagate the same metadata to each
 * checked document.
 */
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import os from 'node:os';
import fs from 'node:fs';
import Database from 'better-sqlite3';

const REPO_ROOT = path.resolve();

interface FreshnessProbeResult {
  hasTopLevelFreshness: boolean;
  topLevelTimestampDefined: boolean;
  topLevelStaleIsBoolean: boolean;
  topLevelSourceIsString: boolean;
  documentHasFreshness: boolean;
}

/**
 * Evaluate a short ESM snippet in a child Node process rooted at the repo root.
 *
 * Used to import `.mjs` implementation modules from a `.ts` test file.
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

/**
 * Build a minimal corpus fixture with one indexed document.
 *
 * The fixture is created in a temporary directory and deleted after the
 * evaluation. A pre-computed freshness proof is supplied to the tool so the
 * test does not depend on filesystem state.
 */
function makeFreshnessFixture(): { databasePath: string; tempDir: string } {
  const tempDir = fs.mkdtempSync(
    path.join(os.tmpdir(), 'freshness-check-red-'),
  );
  const databasePath = path.join(tempDir, 'corpus.sqlite');
  const db = new Database(databasePath);
  try {
    db.exec(fs.readFileSync('./scripts/semantic-index/schema-v2.sql', 'utf8'));
    db.exec(`
      INSERT INTO documents (doc_id, file_path, doc_family, mtime_ms, file_size, sha256, indexed_at)
        VALUES (1, 'src/network.ts', 'ts-source', 1234567890000, 100, 'fixture-sha-001', 1234567890000);
    `);
  } finally {
    db.close();
  }
  return { databasePath, tempDir };
}

describe('freshness-check.mjs freshness transparency', () => {
  describe('freshnessCheck response must carry a freshness proof', () => {
    it('includes a top-level freshness stanza with timestamp, stale flag, and update source', () => {
      const { databasePath, tempDir } = makeFreshnessFixture();
      try {
        const result = runModuleEvaluation<FreshnessProbeResult>(`
          import { freshnessCheck } from './scripts/mcp-semantic/tools/freshness-check.mjs';
          import fs from 'node:fs';

          const databasePath = ${JSON.stringify(databasePath)};
          const response = await freshnessCheck({
            databasePath,
            freshnessProof: {
              mtime_ms: 1234567890000,
              size: 100,
              sha256: 'fixture-sha-001',
            },
          });

          fs.rmSync(${JSON.stringify(tempDir)}, { recursive: true, force: true });

          const freshness = response.freshness;
          console.log(JSON.stringify({
            hasTopLevelFreshness: typeof freshness === 'object' && freshness !== null,
            topLevelTimestampDefined: typeof freshness?.timestamp !== 'undefined',
            topLevelStaleIsBoolean: typeof freshness?.stale === 'boolean',
            topLevelSourceIsString: typeof freshness?.last_update_source === 'string',
            documentHasFreshness: Array.isArray(response.documents) && response.documents.length > 0 && typeof response.documents[0].freshness === 'object' && response.documents[0].freshness !== null,
          }));
        `);

        expect(
          result.hasTopLevelFreshness &&
            result.topLevelTimestampDefined &&
            result.topLevelStaleIsBoolean &&
            result.topLevelSourceIsString,
        ).toBe(true);
      } finally {
        try {
          fs.rmSync(tempDir, { recursive: true, force: true });
        } catch {
          // Ignore best-effort cleanup failures.
        }
      }
    });

    it('propagates freshness metadata to each checked document', () => {
      const { databasePath, tempDir } = makeFreshnessFixture();
      try {
        const result = runModuleEvaluation<FreshnessProbeResult>(`
          import { freshnessCheck } from './scripts/mcp-semantic/tools/freshness-check.mjs';
          import fs from 'node:fs';

          const databasePath = ${JSON.stringify(databasePath)};
          const response = await freshnessCheck({
            databasePath,
            freshnessProof: {
              mtime_ms: 1234567890000,
              size: 100,
              sha256: 'fixture-sha-001',
            },
          });

          fs.rmSync(${JSON.stringify(tempDir)}, { recursive: true, force: true });

          const firstDocument = response.documents?.[0];
          console.log(JSON.stringify({
            hasTopLevelFreshness: typeof response.freshness === 'object' && response.freshness !== null,
            topLevelTimestampDefined: typeof response.freshness?.timestamp !== 'undefined',
            topLevelStaleIsBoolean: typeof response.freshness?.stale === 'boolean',
            topLevelSourceIsString: typeof response.freshness?.last_update_source === 'string',
            documentHasFreshness: typeof firstDocument?.freshness === 'object' && firstDocument?.freshness !== null,
          }));
        `);

        expect(result.documentHasFreshness).toBe(true);
      } finally {
        try {
          fs.rmSync(tempDir, { recursive: true, force: true });
        } catch {
          // Ignore best-effort cleanup failures.
        }
      }
    });
  });
});
