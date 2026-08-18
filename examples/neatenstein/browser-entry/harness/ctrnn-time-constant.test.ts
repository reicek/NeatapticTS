/**
 * Red-phase contract tests for Step B4 item 7:
 * Per-node evolvable time constants (CTRNN formulation).
 *
 * Adds temporal memory to the main agent via CTRNN time constants. The Node
 * class gains a `timeConstant` property (default 1.0), a mutation operator
 * `mutateTimeConstant` perturbs it, and activation uses exponential Euler
 * integration: `state += (input_sum - state) * (dt / timeConstant)`.
 *
 * All tests in this file MUST fail (RED) until the B4 implementation lands.
 *
 * @module
 */

import { describe, expect, it } from '@jest/globals';

import Node from '../../../../src/architecture/node/node';
import * as MutationMethods from '../../../../src/methods/mutation/mutation';

/** Access unknown (future) properties on the mutation methods module. */
const mutation = MutationMethods as Record<string, unknown>;

/** Access unknown (future) properties on Node prototype. */
const nodeProto = Node.prototype as unknown as Record<string, unknown>;

describe('B4.7: Per-node evolvable time constants (CTRNN)', () => {
  describe('B4.7: Node.timeConstant property', () => {
    it('Node instances have a timeConstant property', () => {
      const node = new Node('hidden');
      expect(node).toHaveProperty('timeConstant');
    });

    it('defaults timeConstant to 1.0 (fast response)', () => {
      const node = new Node('hidden');
      expect(node.timeConstant).toBeCloseTo(1.0, 5);
    });

    it('persists timeConstant as a number, not undefined', () => {
      const node = new Node('hidden');
      expect(typeof node.timeConstant).toBe('number');
    });
  });

  describe('B4.7: MOD_TIME_CONSTANT mutation config', () => {
    it('mutation methods export MOD_TIME_CONSTANT config', () => {
      expect(mutation.MOD_TIME_CONSTANT).toBeDefined();
      const config = mutation.MOD_TIME_CONSTANT as { name: string };
      expect(config.name).toBe('MOD_TIME_CONSTANT');
    });

    it('MOD_TIME_CONSTANT is included in the ALL array', () => {
      const all = mutation.ALL as unknown[];
      const names = all.map(
        (c) => (c as { name: string }).name,
      );
      expect(names).toContain('MOD_TIME_CONSTANT');
    });
  });

  describe('B4.7: mutateTimeConstant function', () => {
    it('exports mutateTimeConstant as a function', () => {
      expect(typeof mutation.mutateTimeConstant).toBe('function');
    });

    it('perturbs timeConstant by ±N(0, 0.1) and produces a positive value', () => {
      const node = new Node('hidden');
      const originalTc = node.timeConstant;
      (mutation.mutateTimeConstant as (...a: unknown[]) => void)(
        node,
        () => 0.5,
      );
      expect(node.timeConstant).not.toEqual(originalTc);
      expect(node.timeConstant).toBeGreaterThan(0);
    });

    it('is deterministic for the same RNG seed', () => {
      const nodeA = new Node('hidden');
      const nodeB = new Node('hidden');
      let seedA = 0.42;
      const rngA = () => {
        seedA = (seedA * 9301 + 49297) % 233280;
        return seedA / 233280;
      };
      let seedB = 0.42;
      const rngB = () => {
        seedB = (seedB * 9301 + 49297) % 233280;
        return seedB / 233280;
      };
      (mutation.mutateTimeConstant as (...a: unknown[]) => void)(nodeA, rngA);
      (mutation.mutateTimeConstant as (...a: unknown[]) => void)(nodeB, rngB);
      expect(nodeA.timeConstant).toBeCloseTo(nodeB.timeConstant, 5);
    });
  });

  describe('B4.7: CTRNN exponential Euler integration', () => {
    it('exports applyCtrnnActivation as a function on Node prototype', () => {
      expect(typeof nodeProto.applyCtrnnActivation).toBe('function');
    });

    it('integrates state via exponential Euler: state += (input - state) * (dt / tc)', () => {
      const node = new Node('hidden');
      node.timeConstant = 10.0;
      node.state = 0;
      node.bias = 0;
      // With dt=1 and tc=10, state should move 10% toward the input sum.
      const inputSum = 1.0;
      const dt = 1.0;
      (nodeProto.applyCtrnnActivation as (...a: unknown[]) => void).call(
        node,
        inputSum,
        dt,
      );
      // Expected: state += (1.0 - 0) * (1.0 / 10.0) = 0.1
      expect(node.state).toBeCloseTo(0.1, 5);
    });

    it('with timeConstant=1.0 behaves like direct state assignment (fast response)', () => {
      const node = new Node('hidden');
      node.timeConstant = 1.0;
      node.state = 0;
      node.bias = 0;
      const inputSum = 2.0;
      const dt = 1.0;
      (nodeProto.applyCtrnnActivation as (...a: unknown[]) => void).call(
        node,
        inputSum,
        dt,
      );
      // With tc=1.0 and dt=1.0: state += (2.0 - 0) * 1.0 = 2.0 (full response)
      expect(node.state).toBeCloseTo(2.0, 5);
    });
  });

  describe('B4.7: timeConstant survives serialization', () => {
    it('Node.toJSON includes timeConstant', () => {
      const node = new Node('hidden');
      node.timeConstant = 5.0;
      const json = node.toJSON();
      const jsonRecord = json as Record<string, unknown>;
      expect(jsonRecord.timeConstant).toBe(5.0);
    });

    it('Node.fromJSON restores timeConstant', () => {
      const node = new Node('hidden');
      node.timeConstant = 7.5;
      const json = node.toJSON();
      const restored = Node.fromJSON(json);
      expect(restored.timeConstant).toBeCloseTo(7.5, 5);
    });
  });
});