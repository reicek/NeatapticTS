/**
 * @module freshness.coverage.test
 * @description Coverage tests targeting rag-index/freshness.mjs — exercises
 * getFreshnessProof (the uncovered function) and all isFreshDocument branches.
 */
import { writeFile, mkdir, rm } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  getFreshnessProof,
  isFreshDocument,
} from '../../../rag-index/freshness.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const TEMP_DIR = path.join(__dirname, '.freshness-tmp');

afterAll(async () => {
  await rm(TEMP_DIR, { recursive: true, force: true });
});

describe('freshness — getFreshnessProof', () => {
  it('returns mtime_ms, size, and sha256 for a real file', async () => {
    await mkdir(TEMP_DIR, { recursive: true });
    const filePath = path.join(TEMP_DIR, 'proof.txt');
    const content = 'hello freshness';
    await writeFile(filePath, content, 'utf8');

    const proof = await getFreshnessProof(filePath);

    expect(proof).toEqual({
      mtime_ms: expect.any(Number),
      size: Buffer.byteLength(content, 'utf8'),
      sha256: expect.any(String),
    });
    expect(proof.sha256).toHaveLength(64);
  });

  it('rejects a non-existent file', async () => {
    await expect(
      getFreshnessProof(path.join(TEMP_DIR, 'missing.txt')),
    ).rejects.toThrow();
  });
});

describe('freshness — isFreshDocument', () => {
  const proof = { mtime_ms: 1000, size: 2000, sha256: 'abc' };

  it('returns true when all fields match', () => {
    expect(
      isFreshDocument(
        { mtime_ms: 1000, file_size: 2000, sha256: 'abc' },
        proof,
      ),
    ).toBe(true);
  });

  it('returns false when mtime_ms differs', () => {
    expect(
      isFreshDocument(
        { mtime_ms: 9999, file_size: 2000, sha256: 'abc' },
        proof,
      ),
    ).toBe(false);
  });

  it('returns false when file_size differs', () => {
    expect(
      isFreshDocument(
        { mtime_ms: 1000, file_size: 9999, sha256: 'abc' },
        proof,
      ),
    ).toBe(false);
  });

  it('returns false when sha256 differs', () => {
    expect(
      isFreshDocument(
        { mtime_ms: 1000, file_size: 2000, sha256: 'wrong' },
        proof,
      ),
    ).toBe(false);
  });

  it('returns false when documentRow is null', () => {
    expect(isFreshDocument(null, proof)).toBe(false);
  });

  it('returns false when freshnessProof is null', () => {
    expect(
      isFreshDocument({ mtime_ms: 1000, file_size: 2000, sha256: 'abc' }, null),
    ).toBe(false);
  });

  it('uses file_size from freshnessProof when size is absent', () => {
    const proofWithFileSize = {
      mtime_ms: 1000,
      file_size: 2000,
      sha256: 'abc',
    };
    expect(
      isFreshDocument(
        { mtime_ms: 1000, file_size: 2000, sha256: 'abc' },
        proofWithFileSize,
      ),
    ).toBe(true);
  });
});
