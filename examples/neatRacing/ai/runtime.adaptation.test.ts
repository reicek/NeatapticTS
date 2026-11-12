/**
 * Green contracts for the NeatRacing runtime adaptation engine (slice
 * `P2-S3-A`): recycled capacity caps, per-car seeded adaptation streams, and
 * the sim-side per-tick budget, exercised through the migrated engine.
 *
 * The fixtures follow the recycled racing-curriculum conventions: the AC-026
 * reproducibility tuple drives every per-car random stream through the
 * `'adaptation-<carIndex>'` child-seed table, engine ticks run on seeded twin
 * networks, and the AC-043 budget tests use `jest.useFakeTimers()` to freeze
 * the default clock. One expectation per test, first-principles values.
 *
 * @module
 */

import Network from '../../../src/architecture/network';
import {
  NGE_MAX_EDGE_CAPACITY,
  NGE_MAX_NODE_CAPACITY,
} from '../../../src/neat/nge-juvenile/neat.nge-juvenile.constants';
import {
  createNeatRacingXorShift32Stream,
  deriveNeatRacingAdaptationChildSeed,
  type NeatRacingReproducibilityTuple,
} from '../workers/simulation-worker/reproducibility.utils';
import {
  createNeatRacingPerCarAdaptationEngines,
  createRuntimeAdaptationEngine,
  NEATRACING_ADAPTATION_DEFAULT_LIMITS,
  NEATRACING_ADAPTATION_MAX_CONNECTIONS,
  NEATRACING_ADAPTATION_MAX_NODES,
  NEATRACING_ADAPTATION_TICK_BUDGET_MS,
  resolveNeatRacingAdaptationTickBudget,
  type NeatRacingAdaptationEngineOptions,
  type NeatRacingAdaptationTelemetry,
} from './runtime.adaptation';

/* ------------------------------------------------------------------------- *
 * Fixtures (AC-026 tuple + responsive engine options)
 * ------------------------------------------------------------------------- */

/** AC-026 reproducibility tuple driving the per-car adaptation child seeds. */
const TUPLE: NeatRacingReproducibilityTuple = {
  seed: 42424242,
  agentCount: 3,
  packSchemaVersion: 'test-pack-schema',
  opponentSnapshot: null,
  trackId: 'test-track',
  featureFlags: undefined,
};

/** Rolling evidence window long enough for the recycled 4-entry gate. */
const EVIDENCE_WINDOW = [1, 2, 3, 4];

/** Responsiveness options that remove cadence and cooldown gates per test. */
const RESPONSIVE_ENGINE_OPTIONS: Omit<
  NeatRacingAdaptationEngineOptions,
  'random' | 'carId'
> = {
  cadence: { mode: 'every_tick' },
  improvementThreshold: 0,
  limits: {
    mutationCooldownTicks: 0,
    rollbackCooldownTicks: 0,
  },
};

/** Twin seeded controller network factory (deterministic 6-node baseline). */
function createTwinNetwork(): Network {
  return new Network(4, 2, { seed: 42 });
}

/**
 * Creates an engine whose random stream is derived from the AC-026 tuple
 * child-seed table for `carIndex`, with per-test option overrides on top of
 * the responsive base options.
 *
 * @param carIndex - Car index whose `adaptation-<i>` child seed drives the stream.
 * @param overrides - Optional per-test option overrides.
 * @returns Seeded adaptation engine.
 */
function createSeededEngine(
  carIndex: number,
  overrides: Partial<
    Omit<NeatRacingAdaptationEngineOptions, 'random' | 'carId'>
  > = {},
) {
  return createRuntimeAdaptationEngine({
    ...RESPONSIVE_ENGINE_OPTIONS,
    ...overrides,
    random: createNeatRacingXorShift32Stream(
      deriveNeatRacingAdaptationChildSeed(TUPLE.seed, carIndex),
    ),
  });
}

/**
 * Projects the deterministic decision fields of one telemetry record.
 *
 * @param telemetry - Engine telemetry for one adaptation tick.
 * @returns Decision subset that must replay identically across twin engines.
 */
function pickDecisionFields(telemetry: NeatRacingAdaptationTelemetry): {
  tick: number;
  operations: readonly string[];
  scoreBefore: number;
  scoreAfter: number;
  committed: boolean;
  reason: NeatRacingAdaptationTelemetry['reason'];
} {
  return {
    tick: telemetry.tick,
    operations: [...telemetry.operations],
    scoreBefore: telemetry.scoreBefore,
    scoreAfter: telemetry.scoreAfter,
    committed: telemetry.committed,
    reason: telemetry.reason,
  };
}

describe('NeatRacing runtime adaptation engine (P2-S3-A)', () => {
  // Recycled AC-043 scaffolding: the budget tests freeze the default clock.
  beforeAll(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.runOnlyPendingTimers();
  });

  describe('module surface', () => {
    it('exports createRuntimeAdaptationEngine as a function', () => {
      expect(typeof createRuntimeAdaptationEngine).toBe('function');
    });

    it('exports createNeatRacingPerCarAdaptationEngines as a function', () => {
      expect(typeof createNeatRacingPerCarAdaptationEngines).toBe('function');
    });

    it('exports resolveNeatRacingAdaptationTickBudget as a function', () => {
      expect(typeof resolveNeatRacingAdaptationTickBudget).toBe('function');
    });
  });

  describe('capacity caps (AC-029)', () => {
    it('preserves the recycled 8,000-node capacity', () => {
      expect(NEATRACING_ADAPTATION_MAX_NODES).toBe(8_000);
    });

    it('preserves the recycled 32,000-connection capacity', () => {
      expect(NEATRACING_ADAPTATION_MAX_CONNECTIONS).toBe(32_000);
    });

    it('agrees with the NGE library node capacity', () => {
      expect(NGE_MAX_NODE_CAPACITY).toBe(NEATRACING_ADAPTATION_MAX_NODES);
    });

    it('agrees with the NGE library edge capacity', () => {
      expect(NGE_MAX_EDGE_CAPACITY).toBe(NEATRACING_ADAPTATION_MAX_CONNECTIONS);
    });

    it('keeps the recycled default limits', () => {
      expect(NEATRACING_ADAPTATION_DEFAULT_LIMITS).toMatchObject({
        maxStructuralEditsPerStep: 5,
        maxNodes: 8_000,
        maxConnections: 32_000,
        mutationCooldownTicks: 5,
        rollbackCooldownTicks: 5,
      });
    });

    it('rejects a growth candidate that exceeds the node cap', async () => {
      const engine = createSeededEngine(0, {
        limits: {
          maxNodes: 5,
          mutationCooldownTicks: 0,
          rollbackCooldownTicks: 0,
        },
      });
      const network = createTwinNetwork();

      const telemetry = await engine.adaptOnTick({
        tick: 0,
        network,
        scoreHistory: EVIDENCE_WINDOW,
      });

      expect(telemetry.reason).toBe('safety_checks_failed');
    });

    it('rolls the network back after a failed safety check', async () => {
      const engine = createSeededEngine(0, {
        limits: {
          maxNodes: 5,
          mutationCooldownTicks: 0,
          rollbackCooldownTicks: 0,
        },
      });
      const network = createTwinNetwork();

      await engine.adaptOnTick({
        tick: 0,
        network,
        scoreHistory: EVIDENCE_WINDOW,
      });

      expect(network.nodes.length).toBe(6);
    });
  });

  describe('seeded adaptation streams (D1 / AC-026)', () => {
    it('rejects a missing seeded random stream', () => {
      const optionsWithoutRandom = {
        cadence: { mode: 'every_tick' },
        improvementThreshold: 0,
      } as unknown as NeatRacingAdaptationEngineOptions;

      expect(() => createRuntimeAdaptationEngine(optionsWithoutRandom)).toThrow(
        /random/,
      );
    });

    it('binds one engine per car index without spare factories', () => {
      const engines = createNeatRacingPerCarAdaptationEngines(TUPLE, 2);

      expect(engines.has(0) && engines.has(1) && !engines.has(2)).toBe(true);
    });

    it('replays identical car-0 decisions between the manual and factory engines', async () => {
      const manualEngine = createSeededEngine(0);
      const factoryEngines = createNeatRacingPerCarAdaptationEngines(
        TUPLE,
        1,
        RESPONSIVE_ENGINE_OPTIONS,
      );
      const factoryEngine = factoryEngines.get(0);
      if (!factoryEngine) {
        throw new Error('factory engines must expose car 0');
      }

      const manualTelemetry = await manualEngine.adaptOnTick({
        tick: 0,
        network: createTwinNetwork(),
        scoreHistory: EVIDENCE_WINDOW,
      });
      const factoryTelemetry = await factoryEngine.adaptOnTick({
        tick: 0,
        network: createTwinNetwork(),
        scoreHistory: EVIDENCE_WINDOW,
      });

      expect(pickDecisionFields(manualTelemetry)).toEqual(
        pickDecisionFields(factoryTelemetry),
      );
    });

    it('replays identical car-0 decisions across independent factory instances', async () => {
      const firstEngines = createNeatRacingPerCarAdaptationEngines(
        TUPLE,
        1,
        RESPONSIVE_ENGINE_OPTIONS,
      );
      const secondEngines = createNeatRacingPerCarAdaptationEngines(
        TUPLE,
        1,
        RESPONSIVE_ENGINE_OPTIONS,
      );
      const firstEngine = firstEngines.get(0);
      const secondEngine = secondEngines.get(0);
      if (!firstEngine || !secondEngine) {
        throw new Error('factory engines must expose car 0');
      }

      const firstTelemetry = await firstEngine.adaptOnTick({
        tick: 0,
        network: createTwinNetwork(),
        scoreHistory: EVIDENCE_WINDOW,
      });
      const secondTelemetry = await secondEngine.adaptOnTick({
        tick: 0,
        network: createTwinNetwork(),
        scoreHistory: EVIDENCE_WINDOW,
      });

      expect(pickDecisionFields(firstTelemetry)).toEqual(
        pickDecisionFields(secondTelemetry),
      );
    });
  });

  describe('per-tick adaptation budget (AC-043)', () => {
    it('pins the sim-side tick budget at 2 ms', () => {
      expect(NEATRACING_ADAPTATION_TICK_BUDGET_MS).toBe(2);
    });

    it('accounts a 1 ms tick within the budget', () => {
      const budget = resolveNeatRacingAdaptationTickBudget(100, 101, 2);

      expect(budget.withinBudget).toBe(true);
    });

    it('treats an elapsed time equal to the budget as within', () => {
      const budget = resolveNeatRacingAdaptationTickBudget(100, 102, 2);

      expect(budget.withinBudget).toBe(true);
    });

    it('clamps the residual budget at zero at the budget boundary', () => {
      const budget = resolveNeatRacingAdaptationTickBudget(100, 102, 2);

      expect(budget.residualBudgetMs).toBe(0);
    });

    it('flags a 3.5 ms tick over budget', () => {
      const budget = resolveNeatRacingAdaptationTickBudget(100, 103.5, 2);

      expect(budget.withinBudget).toBe(false);
    });

    it('defaults the budget parameter to NEATRACING_ADAPTATION_TICK_BUDGET_MS', () => {
      const budget = resolveNeatRacingAdaptationTickBudget(100, 100.5);

      expect(budget.budgetMs).toBe(NEATRACING_ADAPTATION_TICK_BUDGET_MS);
    });

    it('stamps zero tick cost for a frozen clock', async () => {
      const engine = createSeededEngine(0, { now: () => 9_000 });
      const network = createTwinNetwork();

      const telemetry = await engine.adaptOnTick({
        tick: 0,
        network,
        scoreHistory: EVIDENCE_WINDOW,
      });

      expect(telemetry.tickCostMs).toBe(0);
    });

    it('stamps a frozen-clock tick within the budget', async () => {
      const engine = createSeededEngine(0, { now: () => 9_000 });
      const network = createTwinNetwork();

      const telemetry = await engine.adaptOnTick({
        tick: 0,
        network,
        scoreHistory: EVIDENCE_WINDOW,
      });

      expect(telemetry.withinTickBudget).toBe(true);
    });

    it('stamps a 10 ms cost for a stepping clock', async () => {
      let steppingClockMs = 0;
      const engine = createSeededEngine(0, {
        now: () => {
          steppingClockMs += 10;
          return steppingClockMs;
        },
      });
      const network = createTwinNetwork();

      const telemetry = await engine.adaptOnTick({
        tick: 0,
        network,
        scoreHistory: EVIDENCE_WINDOW,
      });

      expect(telemetry.tickCostMs).toBe(10);
    });

    it('flags a stepping-clock tick over budget', async () => {
      let steppingClockMs = 0;
      const engine = createSeededEngine(0, {
        now: () => {
          steppingClockMs += 10;
          return steppingClockMs;
        },
      });
      const network = createTwinNetwork();

      const telemetry = await engine.adaptOnTick({
        tick: 0,
        network,
        scoreHistory: EVIDENCE_WINDOW,
      });

      expect(telemetry.withinTickBudget).toBe(false);
    });

    it('stamps a default-clock engine tick within the budget under fake timers', async () => {
      const engine = createSeededEngine(0);
      const network = createTwinNetwork();

      const telemetry = await engine.adaptOnTick({
        tick: 0,
        network,
        scoreHistory: EVIDENCE_WINDOW,
      });

      expect(telemetry.withinTickBudget).toBe(true);
    });

    it('keeps committed decisions clock-independent (D1 replay guard)', async () => {
      const frozenEngine = createSeededEngine(0, { now: () => 9_000 });
      let steppingClockMs = 0;
      const steppingEngine = createSeededEngine(0, {
        now: () => {
          steppingClockMs += 10;
          return steppingClockMs;
        },
      });

      const frozenTelemetry = await frozenEngine.adaptOnTick({
        tick: 0,
        network: createTwinNetwork(),
        scoreHistory: EVIDENCE_WINDOW,
      });
      const steppingTelemetry = await steppingEngine.adaptOnTick({
        tick: 0,
        network: createTwinNetwork(),
        scoreHistory: EVIDENCE_WINDOW,
      });

      expect(steppingTelemetry.reason).toBe(frozenTelemetry.reason);
    });
  });

  describe('recycled growth smoke', () => {
    it('grows a small seeded network under sustained evidence', async () => {
      const engine = createSeededEngine(0);
      const network = createTwinNetwork();
      const initialNodes = network.nodes.length;
      const initialConnections = network.connections.length;

      for (let tick = 0; tick < 60; tick++) {
        await engine.adaptOnTick({
          tick,
          network,
          scoreHistory: EVIDENCE_WINDOW,
        });
      }

      expect(
        network.nodes.length > initialNodes ||
          network.connections.length > initialConnections,
      ).toBe(true);
    });
  });
});
