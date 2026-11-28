/**
 * @module gate-timeout-budgets.test
 * @description Contract tests for the shared Tier-1 gate timeout budgets.
 *
 * The defect these tests lock down: an outer wall-clock ceiling that was not
 * larger than the work it wrapped. When the facade killed a legitimate gate
 * run, the child produced no JSON and the caller reported the opaque
 * "did not return valid JSON" instead of "timed out".
 */
import { describe, expect, it } from '@jest/globals';

import {
  DEFAULT_GATE_TIMEOUT_MS,
  DEFAULT_SUB_GATE_TIMEOUT_MS,
  GATE_MAX_BUFFER_BYTES,
  GATE_TIMEOUT_BUDGETS,
  GATE_TIMEOUT_ENV_VAR,
  GATE_TIMEOUT_OVERHEAD_MS,
  SHARED_VALIDATION_RUNNER_TIMEOUTS,
  SHARED_VALIDATION_RUNNER_TOTAL_MS,
  SHARED_VALIDATION_TIMEOUT_MS,
  SLICE_ADVANCEMENT_SUB_GATES,
  SLICE_ADVANCEMENT_TIMEOUT_MS,
  SUB_GATE_TIMEOUT_BUDGETS,
  describeTimeoutBudgets,
  findGatesWithoutBudget,
  isTimeoutError,
  resolveGateTimeoutMs,
  resolveSubGateTimeoutMs,
} from './gate-timeout-budgets.mjs';

describe('gate-timeout-budgets', () => {
  describe('budget invariants', () => {
    it('sizes the shared-validation budget above the sum of its runner backstops', () => {
      expect(SHARED_VALIDATION_TIMEOUT_MS).toBeGreaterThan(
        SHARED_VALIDATION_RUNNER_TOTAL_MS,
      );
    });

    it('sums the shared-validation runner backstops from the individual budgets', () => {
      const sum =
        SHARED_VALIDATION_RUNNER_TIMEOUTS.tests +
        SHARED_VALIDATION_RUNNER_TIMEOUTS.build +
        SHARED_VALIDATION_RUNNER_TIMEOUTS.lint;
      expect(SHARED_VALIDATION_RUNNER_TOTAL_MS).toBe(sum);
    });

    it('sizes the slice-advancement budget above the sum of its sub-gate budgets', () => {
      const subGateSum = SLICE_ADVANCEMENT_SUB_GATES.reduce(
        (total, gateName) => total + resolveSubGateTimeoutMs(gateName),
        0,
      );
      expect(SLICE_ADVANCEMENT_TIMEOUT_MS).toBeGreaterThan(subGateSum);
      expect(SLICE_ADVANCEMENT_TIMEOUT_MS).toBe(
        subGateSum + GATE_TIMEOUT_OVERHEAD_MS,
      );
    });

    it('gives the facade a slice-advancement budget that covers the consolidated run', () => {
      expect(resolveGateTimeoutMs('slice-advancement')).toBe(
        SLICE_ADVANCEMENT_TIMEOUT_MS,
      );
      expect(resolveGateTimeoutMs('slice-advancement')).toBeGreaterThan(
        SLICE_ADVANCEMENT_SUB_GATES.reduce(
          (total, gateName) => total + resolveSubGateTimeoutMs(gateName),
          0,
        ),
      );
    });

    it('keeps the shared-validation sub-gate budget aligned with the outer budget', () => {
      expect(resolveSubGateTimeoutMs('shared-validation')).toBe(
        SHARED_VALIDATION_TIMEOUT_MS,
      );
    });

    it('gives every declared budget a positive value', () => {
      for (const [gateId, budget] of Object.entries(GATE_TIMEOUT_BUDGETS)) {
        expect(budget).toBeGreaterThan(0);
        expect(Number.isInteger(budget)).toBe(true);
        expect(gateId.length).toBeGreaterThan(0);
      }
    });

    it('never lets a sub-gate budget exceed the consolidated ceiling', () => {
      for (const gateName of SLICE_ADVANCEMENT_SUB_GATES) {
        expect(resolveSubGateTimeoutMs(gateName)).toBeLessThan(
          SLICE_ADVANCEMENT_TIMEOUT_MS,
        );
      }
    });
  });

  describe('budget coverage', () => {
    it('reports no gate without an explicit budget for the facade gate list', async () => {
      const { createGateTools } = await import(
        './mcp/neataptic-gate-mcp.mjs'
      );
      const listGates = createGateTools().find(
        (tool) => tool.name === 'list_gates',
      );
      const payload = await listGates.handler({});
      const gateIds = payload.gates.map((gateDescriptor) => gateDescriptor.id);

      expect(gateIds.length).toBeGreaterThan(0);
      expect(findGatesWithoutBudget(gateIds)).toEqual([]);
    });

    it('reports a gate without a budget so the omission cannot ship silently', () => {
      expect(findGatesWithoutBudget(['plan-sync', 'not-a-real-gate'])).toEqual([
        'not-a-real-gate',
      ]);
    });

    it('falls back to the default budget for an unknown gate', () => {
      expect(resolveGateTimeoutMs('not-a-real-gate')).toBe(
        DEFAULT_GATE_TIMEOUT_MS,
      );
      expect(resolveSubGateTimeoutMs('not-a-real-gate')).toBe(
        DEFAULT_SUB_GATE_TIMEOUT_MS,
      );
    });

    it('describes every budget in the diagnostics report', () => {
      const report = describeTimeoutBudgets();
      expect(Object.keys(report.gateBudgets).sort()).toEqual(
        Object.keys(GATE_TIMEOUT_BUDGETS).sort(),
      );
      expect(Object.keys(report.subGateBudgets).sort()).toEqual(
        Object.keys(SUB_GATE_TIMEOUT_BUDGETS).sort(),
      );
      expect(report.report).toContain('slice-advancement');
    });
  });

  describe('output buffer', () => {
    it('gives every gate spawn an explicit buffer well above the Node default', () => {
      expect(GATE_MAX_BUFFER_BYTES).toBeGreaterThanOrEqual(16 * 1024 * 1024);
    });

    it('reports the buffer limit in the diagnostics report', () => {
      const report = describeTimeoutBudgets();
      expect(report.gateMaxBufferBytes).toBe(GATE_MAX_BUFFER_BYTES);
      expect(report.report).toContain(String(GATE_MAX_BUFFER_BYTES));
    });
  });

  describe('environment override', () => {    it('raises a gate budget when the environment asks for more headroom', () => {
      const raised = resolveGateTimeoutMs('plan-sync', {
        [GATE_TIMEOUT_ENV_VAR]: '900000',
      });
      expect(raised).toBe(900000);
    });

    it('ignores an override that would lower a budget', () => {
      const base = resolveGateTimeoutMs('plan-sync', {});
      expect(resolveGateTimeoutMs('plan-sync', { [GATE_TIMEOUT_ENV_VAR]: '1' })).toBe(
        base,
      );
    });

    it('ignores a malformed or non-positive override', () => {
      const base = resolveGateTimeoutMs('plan-sync', {});
      for (const value of ['', '   ', 'abc', '0', '-5']) {
        expect(
          resolveGateTimeoutMs('plan-sync', { [GATE_TIMEOUT_ENV_VAR]: value }),
        ).toBe(base);
      }
    });

    it('reports the active override in the diagnostics report', () => {
      const report = describeTimeoutBudgets();
      expect(report.envOverrideVar).toBe(GATE_TIMEOUT_ENV_VAR);
      expect(report.envOverrideMs).toBeGreaterThanOrEqual(0);
      expect(report.report).toContain(GATE_TIMEOUT_ENV_VAR);
    });
  });

  describe('timeout detection', () => {
    it('detects the killed flag reported by execFileSync', () => {
      expect(isTimeoutError({ killed: true, signal: null })).toBe(true);
    });

    it('detects a SIGTERM signal', () => {
      expect(isTimeoutError({ signal: 'SIGTERM' })).toBe(true);
    });

    it('detects an ETIMEDOUT error code', () => {
      expect(isTimeoutError({ code: 'ETIMEDOUT' })).toBe(true);
    });

    it('detects an ETIMEDOUT code nested under cause', () => {
      expect(isTimeoutError({ cause: { code: 'ETIMEDOUT' } })).toBe(true);
    });

    it('detects an ETIMEDOUT code on a spawnSync result', () => {
      expect(
        isTimeoutError({
          status: null,
          signal: null,
          error: { code: 'ETIMEDOUT' },
        }),
      ).toBe(true);
    });

    it('does not treat an ordinary crash as a timeout', () => {
      expect(isTimeoutError({ code: 'ENOENT' })).toBe(false);
      expect(isTimeoutError({ signal: 'SIGKILL' })).toBe(false);
      expect(isTimeoutError(null)).toBe(false);
      expect(isTimeoutError(undefined)).toBe(false);
    });
  });
});
