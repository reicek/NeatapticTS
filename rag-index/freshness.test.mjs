import { jest } from '@jest/globals';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { getFreshnessProof, isFreshDocument } from './freshness.mjs';

describe('getFreshnessProof', () => {
  let tempDir;

  beforeEach(() => {
    tempDir = mkdtempSync(path.join(tmpdir(), 'freshness-'));
  });

  afterEach(() => {
    rmSync(tempDir, { recursive: true, force: true });
  });

  it('returns mtime_ms, size, and sha256 for a file', async () => {
    const filePath = path.join(tempDir, 'test.txt');
    writeFileSync(filePath, 'hello world');
    const proof = await getFreshnessProof(filePath);
    expect(proof.size).toBe(11);
    expect(proof.mtime_ms).toBeGreaterThan(0);
    expect(proof.sha256).toHaveLength(64);
  });

  it('computes correct sha256', async () => {
    const filePath = path.join(tempDir, 'hash.txt');
    writeFileSync(filePath, 'test');
    const proof = await getFreshnessProof(filePath);
    // Known SHA-256 of 'test'
    expect(proof.sha256).toBe('9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08');
  });
});

describe('isFreshDocument', () => {
  const baseProof = { mtime_ms: 1000, size: 500, sha256: 'abc123' };

  it('returns true when all fields match', () => {
    expect(isFreshDocument(
      { mtime_ms: 1000, file_size: 500, sha256: 'abc123' },
      baseProof,
    )).toBe(true);
  });

  it('returns false when mtime differs', () => {
    expect(isFreshDocument(
      { mtime_ms: 2000, file_size: 500, sha256: 'abc123' },
      baseProof,
    )).toBe(false);
  });

  it('returns false when file_size differs', () => {
    expect(isFreshDocument(
      { mtime_ms: 1000, file_size: 999, sha256: 'abc123' },
      baseProof,
    )).toBe(false);
  });

  it('returns false when sha256 differs', () => {
    expect(isFreshDocument(
      { mtime_ms: 1000, file_size: 500, sha256: 'different' },
      baseProof,
    )).toBe(false);
  });

  it('returns false when documentRow is null', () => {
    expect(isFreshDocument(null, baseProof)).toBe(false);
  });

  it('returns false when freshnessProof is null', () => {
    expect(isFreshDocument({ mtime_ms: 1000, file_size: 500, sha256: 'abc123' }, null)).toBe(false);
  });

  it('returns false when both are null', () => {
    expect(isFreshDocument(null, null)).toBe(false);
  });

  it('uses file_size fallback when size is not in proof', () => {
    const proofWithFileSize = { mtime_ms: 1000, file_size: 500, sha256: 'abc123' };
    expect(isFreshDocument(
      { mtime_ms: 1000, file_size: 500, sha256: 'abc123' },
      proofWithFileSize,
    )).toBe(true);
  });

  it('handles string mtime_ms from DB row', () => {
    expect(isFreshDocument(
      { mtime_ms: '1000', file_size: 500, sha256: 'abc123' },
      baseProof,
    )).toBe(true);
  });

  it('handles string file_size from DB row', () => {
    expect(isFreshDocument(
      { mtime_ms: 1000, file_size: '500', sha256: 'abc123' },
      baseProof,
    )).toBe(true);
  });

  it('returns false when both size and file_size are undefined in proof', () => {
    const proofNoSize = { mtime_ms: 1000, sha256: 'abc123' };
    expect(isFreshDocument(
      { mtime_ms: 1000, file_size: 500, sha256: 'abc123' },
      proofNoSize,
    )).toBe(false);
  });
});