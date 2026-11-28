/**
 * @module slice-advancement-aggregation.test
 * @description Contract tests for the consolidated gate's result aggregation.
 *
 * The rule under test: a sub-gate that TIMED OUT never produced a verdict,
 * so the slice it covers was never validated. Reporting such a slice as
 * green would be a false green, so a timeout must fail the consolidated
 * gate — unlike an ordinary tooling error, which is reported and skipped.
 */
import { describe, expect, it } from '@jest/globals';

import { summarizeSubGateResults } from './slice-advancement-aggregation.mjs';

function passingResult(gate) {
  return {
    gate,
    pass: true,
    evidence: {},
    fixHint: null,
    gate_error: false,
    timed_out: false,
    timeoutMs: 60_000,
  };
}

function failingResult(gate) {
  return { ...passingResult(gate), pass: false, fixHint: 'fix it' };
}

function erroredResult(gate) {
  return {
    ...passingResult(gate),
    pass: false,
    gate_error: true,
    fixHint: 'tooling failure',
    evidence: { error: 'boom' },
  };
}

function timedOutResult(gate) {
  return {
    ...passingResult(gate),
    pass: false,
    gate_error: true,
    timed_out: true,
    fixHint: `Sub-gate '${gate}' exceeded its budget.`,
    evidence: { error: 'killed', timeoutMs: 60_000, timedOut: true },
  };
}

describe('slice-advancement aggregation', () => {
  it('passes when every sub-gate passes', () => {
    const summary = summarizeSubGateResults([
      passingResult('plan-sync'),
      passingResult('shared-validation'),
    ]);

    expect(summary.allPassed).toBe(true);
    expect(summary.failedGates).toEqual([]);
    expect(summary.timedOutGates).toEqual([]);
    expect(summary.sub_gates).toHaveLength(2);
  });

  it('fails when a sub-gate reports a content failure', () => {
    const summary = summarizeSubGateResults([
      passingResult('plan-sync'),
      failingResult('shared-validation'),
    ]);

    expect(summary.allPassed).toBe(false);
    expect(summary.failedGates).toEqual(['shared-validation']);
  });

  it('skips a tooling error without failing the consolidated gate', () => {
    const summary = summarizeSubGateResults([
      passingResult('plan-sync'),
      erroredResult('code-coverage'),
    ]);

    expect(summary.allPassed).toBe(true);
    expect(summary.erroredGates).toEqual(['code-coverage']);
    expect(summary.failedGates).toEqual([]);
  });

  it('fails the consolidated gate when a sub-gate timed out', () => {
    const summary = summarizeSubGateResults([
      passingResult('plan-sync'),
      timedOutResult('shared-validation'),
    ]);

    expect(summary.allPassed).toBe(false);
    expect(summary.timedOutGates).toEqual(['shared-validation']);
    expect(summary.failedGates).toEqual([]);
  });

  it('records the applied timeout budget for every sub-gate', () => {
    const summary = summarizeSubGateResults([
      passingResult('plan-sync'),
      timedOutResult('shared-validation'),
    ]);

    expect(summary.sub_gates[0]).toMatchObject({
      name: 'plan-sync',
      timed_out: false,
      timeout_ms: 60_000,
    });
    expect(summary.sub_gates[1]).toMatchObject({
      name: 'shared-validation',
      timed_out: true,
      timeout_ms: 60_000,
      error: 'killed',
    });
  });

  it('treats an empty result set as passing', () => {
    const summary = summarizeSubGateResults([]);
    expect(summary.allPassed).toBe(true);
    expect(summary.sub_gates).toEqual([]);
  });
});
