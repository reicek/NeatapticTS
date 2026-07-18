import {
  MorphDeltaRegistry,
  validateMorphDelta,
} from './neat.nge-juvenile.grow';
import {
  NgeJuvenile_BudgetError,
  NgeJuvenile_MorphError,
} from './neat.nge-juvenile.errors';
import type { NgeGrowthBudget, NgeMorphDelta } from './neat.nge-juvenile.types';

describe('MorphDeltaRegistry', () => {
  const baseBudget: NgeGrowthBudget = {
    maxNodes: 10,
    maxEdges: 10,
    maxEpisodicSlots: 10,
    currentNodeCount: 5,
    currentEdgeCount: 5,
    currentEpisodicSlotCount: 5,
  };

  const makeDelta = (kind: NgeMorphDelta['kind']): NgeMorphDelta => ({
    kind,
    targetModuleId: 'test:module',
    detail: {},
    wiringCostDelta: 0,
  });

  it('validates an edgeDensify delta that fits the edge budget', () => {
    const delta: NgeMorphDelta = {
      ...makeDelta('edgeDensify'),
      wiringCostDelta: 1,
    };
    expect(() =>
      validateMorphDelta(delta, { ...baseBudget, currentEdgeCount: 5 }),
    ).not.toThrow();
  });

  it('throws a budget error when edgeDensify would exceed the edge budget', () => {
    const delta: NgeMorphDelta = {
      ...makeDelta('edgeDensify'),
      wiringCostDelta: 10,
    };
    expect(() =>
      validateMorphDelta(delta, { ...baseBudget, currentEdgeCount: 5 }),
    ).toThrow(NgeJuvenile_BudgetError);
  });

  it('validates a nodeAdd delta that fits the node budget', () => {
    const delta: NgeMorphDelta = {
      ...makeDelta('nodeAdd'),
      detail: { proposedAdditions: 2 },
      wiringCostDelta: 0,
    };
    expect(() =>
      validateMorphDelta(delta, { ...baseBudget, currentNodeCount: 5 }),
    ).not.toThrow();
  });

  it('throws a budget error when nodeAdd would exceed the node budget', () => {
    const delta: NgeMorphDelta = {
      ...makeDelta('nodeAdd'),
      detail: { proposedAdditions: 10 },
      wiringCostDelta: 0,
    };
    expect(() =>
      validateMorphDelta(delta, { ...baseBudget, currentNodeCount: 5 }),
    ).toThrow(NgeJuvenile_BudgetError);
  });

  it('validates a slotExpand delta that fits the episodic slot budget', () => {
    const delta: NgeMorphDelta = {
      ...makeDelta('slotExpand'),
      wiringCostDelta: 1,
    };
    expect(() =>
      validateMorphDelta(delta, {
        ...baseBudget,
        currentEpisodicSlotCount: 5,
      }),
    ).not.toThrow();
  });

  it('throws a budget error when slotExpand would exceed the episodic slot budget', () => {
    const delta: NgeMorphDelta = {
      ...makeDelta('slotExpand'),
      wiringCostDelta: 10,
    };
    expect(() =>
      validateMorphDelta(delta, {
        ...baseBudget,
        currentEpisodicSlotCount: 5,
      }),
    ).toThrow(NgeJuvenile_BudgetError);
  });

  it('always validates edgePrune and compact deltas', () => {
    expect(() =>
      validateMorphDelta(makeDelta('edgePrune'), baseBudget),
    ).not.toThrow();
    expect(() =>
      validateMorphDelta(makeDelta('compact'), baseBudget),
    ).not.toThrow();
  });

  it('throws a morph error for unregistered kinds', () => {
    MorphDeltaRegistry.unregister('edgePrune');
    expect(() =>
      validateMorphDelta(makeDelta('edgePrune'), baseBudget),
    ).toThrow(NgeJuvenile_MorphError);
    MorphDeltaRegistry.register('edgePrune', () => {
      // Pruning frees structural budget; no validation needed.
    });
  });

  it('allows runtime registration of custom validators', () => {
    MorphDeltaRegistry.register('custom', () => {
      throw new NgeJuvenile_MorphError('custom failure');
    });
    expect(() =>
      validateMorphDelta(
        {
          kind: 'custom',
          targetModuleId: 'm',
          detail: {},
          wiringCostDelta: 0,
        } as unknown as NgeMorphDelta,
        baseBudget,
      ),
    ).toThrow(NgeJuvenile_MorphError);
    MorphDeltaRegistry.unregister('custom');
  });
});
