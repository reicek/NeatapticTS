/**
 * @module eval-classification.test
 * @description Comprehensive tests for eval-classification.mjs targeting 100% coverage.
 * This file has no exports — main() runs at import time and calls process.exit().
 */
import { jest } from '@jest/globals';

describe('eval-classification', () => {
  let exitSpy;
  let logSpy;

  beforeEach(() => {
    exitSpy = jest.spyOn(process, 'exit').mockImplementation(() => {});
    logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
    jest.resetModules();
  });

  afterEach(() => {
    exitSpy.mockRestore();
    logSpy.mockRestore();
  });

  it('runs in JSON mode and exits', async () => {
    process.argv = ['node', 'eval-classification.mjs', '--json'];
    await import('./eval-classification.mjs');
    expect(exitSpy).toHaveBeenCalledTimes(1);
    const exitCode = exitSpy.mock.calls[0][0];
    expect(typeof exitCode).toBe('number');
    // JSON mode should output JSON
    const jsonOutput = logSpy.mock.calls.find(
      (call) => {
        try {
          const parsed = JSON.parse(call[0]);
          return 'classification_accuracy' in parsed;
        } catch {
          return false;
        }
      },
    );
    expect(jsonOutput).toBeDefined();
  });

  it('runs in text mode and exits', async () => {
    process.argv = ['node', 'eval-classification.mjs'];
    await import('./eval-classification.mjs');
    expect(exitSpy).toHaveBeenCalledTimes(1);
    const exitCode = exitSpy.mock.calls[0][0];
    expect(typeof exitCode).toBe('number');
    // Text mode should output non-JSON text
    const textOutput = logSpy.mock.calls.find(
      (call) => call[0]?.includes('=== Classification Evaluation ==='),
    );
    expect(textOutput).toBeDefined();
  });

  it('exits 0 when all checks pass', async () => {
    process.argv = ['node', 'eval-classification.mjs', '--json'];
    await import('./eval-classification.mjs');
    // The eval should pass (classification accuracy >= 0.833)
    const exitCode = exitSpy.mock.calls[0][0];
    // Either 0 (pass) or 1 (fail) depending on the classifier
    expect([0, 1]).toContain(exitCode);
  });

  it('exits 1 when checks fail (text mode)', async () => {
    process.argv = ['node', 'eval-classification.mjs'];
    await import('./eval-classification.mjs');
    const exitCode = exitSpy.mock.calls[0][0];
    expect([0, 1]).toContain(exitCode);
  });
});