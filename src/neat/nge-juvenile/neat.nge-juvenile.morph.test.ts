/**
 * Red tests for the NGE juvenile morph-delta registry / dispatch refactor.
 *
 * `validateMorphDelta` currently implements morph validation as a closed
 * switch statement over `NgeMorphDelta.kind`. The planned refactor replaces that
 * switch with a `MorphDeltaRegistry` dispatch table that supports:
 *
 *   - register(kind, validator)
 *   - validate(delta, budget)
 *   - unregister(kind)
 *
 * These tests exercise the registry contract directly and confirm that
 * `validateMorphDelta` delegates to it. They should fail until the registry
 * is implemented and wired into `validateMorphDelta`.
 */

import { NgeJuvenile_MorphError } from './neat.nge-juvenile.errors';
import {
  MorphDeltaRegistry,
  validateMorphDelta,
} from './neat.nge-juvenile.grow';
import type { NgeGrowthBudget, NgeMorphDelta } from './neat.nge-juvenile.types';

/**
 * Budget fixture with generous headroom for every growth dimension.
 */
function makePermissiveBudget(): NgeGrowthBudget {
  return {
    maxNodes: 100,
    maxEdges: 100,
    maxEpisodicSlots: 100,
    currentNodeCount: 0,
    currentEdgeCount: 0,
    currentEpisodicSlotCount: 0,
  };
}

/**
 * Minimal delta fixture. The kind is cast because the red phase must be able
 * to create test-only morph kinds that are not yet part of the closed union.
 */
function makeDelta(
  kind: string,
  overrides: Partial<NgeMorphDelta> = {},
): NgeMorphDelta {
  return {
    kind: kind as NgeMorphDelta['kind'],
    targetModuleId: 'module-A',
    detail: {},
    wiringCostDelta: 0,
    ...overrides,
  };
}

describe('nge juvenile morph delta registry', () => {
  describe('MorphDeltaRegistry', () => {
    it('exports an object with register, validate, and unregister methods', () => {
      expect(MorphDeltaRegistry).toEqual(
        expect.objectContaining({
          register: expect.any(Function),
          validate: expect.any(Function),
          unregister: expect.any(Function),
        }),
      );
    });
  });

  describe('validate', () => {
    it('passes for edgeDensify when the edge budget has headroom', () => {
      // Arrange
      const delta = makeDelta('edgeDensify', { wiringCostDelta: 1 });
      const budget = makePermissiveBudget();

      // Act & Assert
      expect(() => MorphDeltaRegistry.validate(delta, budget)).not.toThrow();
    });

    it('passes for slotExpand when the episodic slot budget has headroom', () => {
      // Arrange
      const delta = makeDelta('slotExpand', { wiringCostDelta: 1 });
      const budget = makePermissiveBudget();

      // Act & Assert
      expect(() => MorphDeltaRegistry.validate(delta, budget)).not.toThrow();
    });

    it('passes for nodeAdd when the node budget has headroom', () => {
      // Arrange
      const delta = makeDelta('nodeAdd', {
        detail: { proposedAdditions: 1 },
      });
      const budget = makePermissiveBudget();

      // Act & Assert
      expect(() => MorphDeltaRegistry.validate(delta, budget)).not.toThrow();
    });

    it('passes for edgePrune without inspecting the budget', () => {
      // Arrange
      const delta = makeDelta('edgePrune');
      const budget = makePermissiveBudget();

      // Act & Assert
      expect(() => MorphDeltaRegistry.validate(delta, budget)).not.toThrow();
    });

    it('passes for compact without inspecting the budget', () => {
      // Arrange
      const delta = makeDelta('compact');
      const budget = makePermissiveBudget();

      // Act & Assert
      expect(() => MorphDeltaRegistry.validate(delta, budget)).not.toThrow();
    });
  });

  describe('register', () => {
    it('accepts a new test-only kind without throwing', () => {
      // Arrange
      const validator = jest.fn();

      // Act & Assert
      expect(() =>
        MorphDeltaRegistry.register('testOnlyKind', validator),
      ).not.toThrow();
    });

    it('makes a registered test-only kind pass validation', () => {
      // Arrange
      const validator = jest.fn();
      MorphDeltaRegistry.register('testOnlyKind', validator);
      const delta = makeDelta('testOnlyKind');
      const budget = makePermissiveBudget();

      // Act & Assert
      expect(() => MorphDeltaRegistry.validate(delta, budget)).not.toThrow();
    });
  });

  describe('unregister', () => {
    it('removes a previously registered test-only kind so validation throws', () => {
      // Arrange
      const validator = jest.fn();
      MorphDeltaRegistry.register('testOnlyKind', validator);
      MorphDeltaRegistry.unregister('testOnlyKind');
      const delta = makeDelta('testOnlyKind');
      const budget = makePermissiveBudget();

      // Act & Assert
      expect(() => MorphDeltaRegistry.validate(delta, budget)).toThrow(
        NgeJuvenile_MorphError,
      );
    });
  });

  describe('validateMorphDelta', () => {
    it('delegates to MorphDeltaRegistry.validate', () => {
      // Arrange
      const registrySpy = jest.spyOn(MorphDeltaRegistry, 'validate');
      const delta = makeDelta('edgeDensify', { wiringCostDelta: 1 });
      const budget = makePermissiveBudget();

      // Act
      validateMorphDelta(delta, budget);

      // Assert
      expect(registrySpy).toHaveBeenCalledWith(delta, budget);

      // Cleanup
      registrySpy.mockRestore();
    });
  });
});
