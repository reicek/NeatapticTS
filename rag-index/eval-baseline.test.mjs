/**
 * @module eval-baseline.test
 * @description Comprehensive tests for eval-baseline.mjs targeting 100% coverage.
 */

import { jest } from '@jest/globals';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const realFsPromises = require('fs/promises');

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const mockMkdir = jest.fn();
const mockReadFile = jest.fn();
const mockWriteFile = jest.fn();

jest.unstable_mockModule('node:fs/promises', () => ({
  mkdir: mockMkdir,
  readFile: mockReadFile,
  writeFile: mockWriteFile,
}));

const { storeBaseline, loadBaseline, compareToBaseline } = await import('./eval-baseline.mjs');

// ---------------------------------------------------------------------------
// storeBaseline
// ---------------------------------------------------------------------------

describe('storeBaseline', () => {
  beforeEach(() => {
    mockMkdir.mockReset();
    mockWriteFile.mockReset();
    mockMkdir.mockResolvedValue(undefined);
    mockWriteFile.mockResolvedValue(undefined);
  });

  it('stores a baseline and returns it', async () => {
    const data = {
      baseline_id: 'baseline-test-001',
      conditions: { hybrid: { mrr_at_5: 0.32 } },
    };
    const result = await storeBaseline(data);
    expect(result).toEqual(data);
    expect(mockMkdir).toHaveBeenCalledTimes(1);
    expect(mockWriteFile).toHaveBeenCalledTimes(1);
    const writtenContent = mockWriteFile.mock.calls[0][1];
    expect(JSON.parse(writtenContent)).toEqual(data);
  });

  it('uses custom path when provided (relative)', async () => {
    const data = { baseline_id: 'b1', conditions: {} };
    await storeBaseline(data, { path: 'custom/baseline.json' });
    const filePath = mockWriteFile.mock.calls[0][0];
    expect(filePath).toContain('custom');
  });

  it('uses custom path when provided (absolute)', async () => {
    const data = { baseline_id: 'b1', conditions: {} };
    const absPath = path.join(__dirname, 'abs-baseline.json');
    await storeBaseline(data, { path: absPath });
    const filePath = mockWriteFile.mock.calls[0][0];
    expect(filePath).toBe(absPath);
  });

  it('throws when baseline_id is missing', async () => {
    await expect(storeBaseline({ conditions: {} })).rejects.toThrow(
      'baseline_id is required to store a baseline.',
    );
  });

  it('throws when data is null', async () => {
    await expect(storeBaseline(null)).rejects.toThrow(
      'baseline_id is required to store a baseline.',
    );
  });

  it('throws when baseline_id is empty string', async () => {
    await expect(storeBaseline({ baseline_id: '', conditions: {} })).rejects.toThrow(
      'baseline_id is required to store a baseline.',
    );
  });

  it('throws when baseline_id is not a string', async () => {
    await expect(storeBaseline({ baseline_id: 123, conditions: {} })).rejects.toThrow(
      'baseline_id is required to store a baseline.',
    );
  });
});

// ---------------------------------------------------------------------------
// loadBaseline
// ---------------------------------------------------------------------------

describe('loadBaseline', () => {
  beforeEach(() => {
    mockReadFile.mockReset();
  });

  it('loads and parses a baseline file', async () => {
    const baselineData = { baseline_id: 'b1', conditions: {} };
    mockReadFile.mockResolvedValue(JSON.stringify(baselineData));
    const result = await loadBaseline('b1');
    expect(result).toEqual(baselineData);
  });

  it('uses custom path when provided', async () => {
    const baselineData = { baseline_id: 'b1', conditions: {} };
    mockReadFile.mockResolvedValue(JSON.stringify(baselineData));
    await loadBaseline('b1', { path: 'custom/path.json' });
    const filePath = mockReadFile.mock.calls[0][0];
    expect(filePath).toContain('custom');
  });

  it('throws when baselineId is not a string', async () => {
    await expect(loadBaseline(123)).rejects.toThrow(
      'baseline_id is required to load a baseline.',
    );
  });

  it('throws when baselineId is empty string', async () => {
    await expect(loadBaseline('')).rejects.toThrow(
      'baseline_id is required to load a baseline.',
    );
  });

  it('throws when file cannot be read', async () => {
    mockReadFile.mockRejectedValue(new Error('ENOENT'));
    await expect(loadBaseline('nonexistent')).rejects.toThrow('ENOENT');
  });
});

// ---------------------------------------------------------------------------
// compareToBaseline
// ---------------------------------------------------------------------------

describe('compareToBaseline', () => {
  const baseQuery = { condition: 'hybrid', metrics: { mrr_at_5: 0.3 } };

  it('returns PASS when current meets baseline', () => {
    const report = compareToBaseline(
      { condition: 'hybrid', metrics: { mrr_at_5: 0.3 } },
      { condition: 'hybrid', metrics: { mrr_at_5: 0.3 } },
      { failThresholds: { mrr_at_5: 0.01 } },
    );
    expect(report.pass).toBe(true);
    expect(report.metrics.mrr_at_5.status).toBe('PASS');
  });

  it('returns FAIL when regression exceeds failThreshold', () => {
    const report = compareToBaseline(
      { condition: 'hybrid', metrics: { mrr_at_5: 0.28 } },
      { condition: 'hybrid', metrics: { mrr_at_5: 0.32 } },
      { failThresholds: { mrr_at_5: 0.01 } },
    );
    expect(report.pass).toBe(false);
    expect(report.metrics.mrr_at_5.status).toBe('FAIL');
    expect(report.regressions).toContain('mrr_at_5');
  });

  it('returns WARN when regression exceeds warnThreshold but not failThreshold', () => {
    const report = compareToBaseline(
      { condition: 'hybrid', metrics: { mrr_at_5: 0.29 } },
      { condition: 'hybrid', metrics: { mrr_at_5: 0.32 } },
      {
        failThresholds: { mrr_at_5: 0.05 },
        warnThresholds: { mrr_at_5: 0.01 },
      },
    );
    expect(report.pass).toBe(true);
    expect(report.metrics.mrr_at_5.status).toBe('WARN');
    expect(report.warnings).toContain('mrr_at_5');
  });

  it('uses default latencyWarnMs of 100', () => {
    const report = compareToBaseline(
      { condition: 'hybrid', metrics: { latency_ms: { p95: 250 } } },
      { condition: 'hybrid', metrics: { latency_ms: { p95: 100 } } },
      {},
    );
    expect(report.metrics.latency_p95.status).toBe('WARN');
    expect(report.warnings).toContain('latency_p95');
  });

  it('reports PASS for latency when within threshold', () => {
    const report = compareToBaseline(
      { condition: 'hybrid', metrics: { latency_ms: { p95: 150 } } },
      { condition: 'hybrid', metrics: { latency_ms: { p95: 100 } } },
      { latencyWarnMs: 100 },
    );
    expect(report.metrics.latency_p95.status).toBe('PASS');
  });

  it('handles latency_ms as a number (not object)', () => {
    const report = compareToBaseline(
      { condition: 'hybrid', metrics: { latency_ms: 200 } },
      { condition: 'hybrid', metrics: { latency_ms: 100 } },
      { latencyWarnMs: 50 },
    );
    expect(report.metrics.latency_p95.status).toBe('WARN');
    expect(report.metrics.latency_p95.current).toBe(200);
    expect(report.metrics.latency_p95.baseline).toBe(100);
  });

  it('handles latency_ms as null/undefined (defaults to 0)', () => {
    const report = compareToBaseline(
      { condition: 'hybrid', metrics: {} },
      { condition: 'hybrid', metrics: {} },
      {},
    );
    expect(report.metrics.latency_p95.current).toBe(0);
    expect(report.metrics.latency_p95.baseline).toBe(0);
    expect(report.metrics.latency_p95.status).toBe('PASS');
  });

  it('handles latency_ms object with null p95', () => {
    const report = compareToBaseline(
      { condition: 'hybrid', metrics: { latency_ms: { p95: null } } },
      { condition: 'hybrid', metrics: { latency_ms: { p95: null } } },
      {},
    );
    expect(report.metrics.latency_p95.current).toBe(0);
    expect(report.metrics.latency_p95.baseline).toBe(0);
  });

  it('handles warn-only metric (not in failThresholds)', () => {
    const report = compareToBaseline(
      { condition: 'hybrid', metrics: { ndcg_at_5: 0.28 } },
      { condition: 'hybrid', metrics: { ndcg_at_5: 0.32 } },
      { warnThresholds: { ndcg_at_5: 0.01 } },
    );
    expect(report.metrics.ndcg_at_5.status).toBe('WARN');
    expect(report.pass).toBe(true);
  });

  it('handles metric in both fail and warn thresholds (fail takes priority)', () => {
    const report = compareToBaseline(
      { condition: 'hybrid', metrics: { mrr_at_5: 0.20 } },
      { condition: 'hybrid', metrics: { mrr_at_5: 0.32 } },
      {
        failThresholds: { mrr_at_5: 0.01 },
        warnThresholds: { mrr_at_5: 0.005 },
      },
    );
    expect(report.metrics.mrr_at_5.status).toBe('FAIL');
  });

  it('handles positive delta (improvement) as PASS', () => {
    const report = compareToBaseline(
      { condition: 'hybrid', metrics: { mrr_at_5: 0.40 } },
      { condition: 'hybrid', metrics: { mrr_at_5: 0.30 } },
      { failThresholds: { mrr_at_5: 0.01 }, warnThresholds: { mrr_at_5: 0.01 } },
    );
    expect(report.metrics.mrr_at_5.status).toBe('PASS');
    expect(report.metrics.mrr_at_5.delta).toMatch(/^\+/);
  });

  it('handles negative delta formatting', () => {
    const report = compareToBaseline(
      { condition: 'hybrid', metrics: { mrr_at_5: 0.28 } },
      { condition: 'hybrid', metrics: { mrr_at_5: 0.30 } },
      { failThresholds: { mrr_at_5: 0.05 } },
    );
    expect(report.metrics.mrr_at_5.delta).toMatch(/^-/);
  });

  it('throws when current is null', () => {
    expect(() => compareToBaseline(null, baseQuery)).toThrow(
      'current run object is required.',
    );
  });

  it('throws when current is not an object', () => {
    expect(() => compareToBaseline('string', baseQuery)).toThrow(
      'current run object is required.',
    );
  });

  it('throws when baseline is null', () => {
    expect(() => compareToBaseline(baseQuery, null)).toThrow(
      'baseline object is required.',
    );
  });

  it('throws when baseline is not an object', () => {
    expect(() => compareToBaseline(baseQuery, 'string')).toThrow(
      'baseline object is required.',
    );
  });

  it('throws on condition mismatch', () => {
    expect(() =>
      compareToBaseline(
        { condition: 'hybrid', metrics: {} },
        { condition: 'bm25_only', metrics: {} },
      ),
    ).toThrow('Condition mismatch');
  });

  it('throws when baseline.metrics is missing', () => {
    expect(() =>
      compareToBaseline(
        { condition: 'hybrid', metrics: {} },
        { condition: 'hybrid' },
      ),
    ).toThrow('baseline metrics are required.');
  });

  it('throws when baseline.metrics is not an object', () => {
    expect(() =>
      compareToBaseline(
        { condition: 'hybrid', metrics: {} },
        { condition: 'hybrid', metrics: 'notobject' },
      ),
    ).toThrow('baseline metrics are required.');
  });

  it('throws when current.metrics is missing', () => {
    expect(() =>
      compareToBaseline(
        { condition: 'hybrid' },
        { condition: 'hybrid', metrics: {} },
      ),
    ).toThrow('current metrics are required.');
  });

  it('throws when current.metrics is not an object', () => {
    expect(() =>
      compareToBaseline(
        { condition: 'hybrid', metrics: 'notobject' },
        { condition: 'hybrid', metrics: {} },
      ),
    ).toThrow('current metrics are required.');
  });

  it('handles metrics with undefined values (defaults to 0)', () => {
    const report = compareToBaseline(
      { condition: 'hybrid', metrics: {} },
      { condition: 'hybrid', metrics: {} },
      { failThresholds: { mrr_at_5: 0.01 } },
    );
    expect(report.metrics.mrr_at_5.current).toBe(0);
    expect(report.metrics.mrr_at_5.baseline).toBe(0);
    expect(report.metrics.mrr_at_5.status).toBe('PASS');
  });

  it('returns condition in the report', () => {
    const report = compareToBaseline(
      { condition: 'hybrid', metrics: {} },
      { condition: 'hybrid', metrics: {} },
    );
    expect(report.condition).toBe('hybrid');
  });

  it('handles custom latencyWarnMs', () => {
    const report = compareToBaseline(
      { condition: 'hybrid', metrics: { latency_ms: { p95: 350 } } },
      { condition: 'hybrid', metrics: { latency_ms: { p95: 100 } } },
      { latencyWarnMs: 200 },
    );
    expect(report.metrics.latency_p95.status).toBe('WARN');
  });
});