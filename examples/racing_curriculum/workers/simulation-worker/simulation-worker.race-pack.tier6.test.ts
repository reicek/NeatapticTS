/**
 * Red-phase contracts for Tier 6 hall-of-fame wiring and snapshot type adapter.
 *
 * Contracts verified here:
 * - OpponentSnapshotPool is exposed in protocol state after a race completes
 * - Opponent snapshots accumulate across multiple generations
 * - A type adapter function converts core OpponentSnapshot to race-pack shape
 *
 * All tests stay red until Step 04 wires the hall-of-fame snapshot pool into
 * the racing coevolution loop and adds the type adapter.
 *
 * Single-expect rule is enforced throughout.
 */
import {
  routeRacingWorkerProtocolMessage,
  createInitialProtocolState,
} from './simulation-worker.evolution.protocol.service';
import type { EvolutionProtocolState } from './simulation-worker.evolution.types';
import type { OpponentSnapshotPool } from '../../../../src/neat/nge-collective/neat.nge-collective';

// ---------------------------------------------------------------------------
// Mock race episode runner — simulates a race that is already finished
// ---------------------------------------------------------------------------

interface MockRaceRunner {
  tick(): void;
  frame: { done: true };
  serializeVisualizationNetwork(): Float32Array;
}

function createDoneMockRunner(): MockRaceRunner {
  return {
    tick: () => {},
    frame: { done: true },
    serializeVisualizationNetwork: () => new Float32Array([0]),
  };
}

// ---------------------------------------------------------------------------
// Helper: walk the FSM to a racing state, then swap in a done runner
// ---------------------------------------------------------------------------

function createRacingStateWithDoneRunner(): EvolutionProtocolState {
  const initResult = routeRacingWorkerProtocolMessage(
    { type: 'init', populationSize: 10, rngSeed: 42, tier: 6 },
    createInitialProtocolState(),
  );
  const genResult = routeRacingWorkerProtocolMessage(
    { type: 'request-generation' },
    initResult.nextState,
  );
  const raceResult = routeRacingWorkerProtocolMessage(
    { type: 'start-race', tierConfig: null, opponentSnapshotId: 'test' },
    genResult.nextState,
  );
  return {
    ...raceResult.nextState,
    raceRunner: createDoneMockRunner(),
  };
}

// ---------------------------------------------------------------------------
// Dynamic import for snapshot type adapter (does not exist yet — Step 04)
// ---------------------------------------------------------------------------

/** Race-pack-local opponent snapshot shape (from race-pack.service.ts). */
interface RacePackOpponentSnapshot {
  readonly snapshotId: string;
  readonly generation: number;
  readonly networkPayloads: readonly unknown[];
}

/** Core opponent snapshot shape (from nge-collective.types.ts). */
interface CoreOpponentSnapshot {
  agentId: string;
  snapshot: Readonly<Record<string, unknown>>;
  frozenAt: number;
}

interface SnapshotAdapterModule {
  convertCoreToRacePackSnapshot(
    core: CoreOpponentSnapshot,
  ): RacePackOpponentSnapshot;
}

async function loadSnapshotAdapter(): Promise<SnapshotAdapterModule> {
  const modulePath = './simulation-worker.race-pack.service';
  const module = (await import(modulePath)) as Partial<SnapshotAdapterModule>;

  if (typeof module.convertCoreToRacePackSnapshot !== 'function') {
    throw new Error(
      'Missing race-pack service export: convertCoreToRacePackSnapshot',
    );
  }

  return module as SnapshotAdapterModule;
}

// ---------------------------------------------------------------------------
// Red tests — Hall-of-fame opponent snapshot pool wiring
// ---------------------------------------------------------------------------

describe('Tier 6 hall-of-fame opponent snapshot pool wiring', () => {
  it('exposes an OpponentSnapshotPool in protocol state after race completes', () => {
    // Arrange — racing state with a mock runner that reports done
    const racingState = createRacingStateWithDoneRunner();

    // Act — send request-race-step
    const result = routeRacingWorkerProtocolMessage(
      { type: 'request-race-step', requestId: 'hof-1', stepsToAdvance: 1 },
      racingState,
    );

    // Assert — protocol state should contain an OpponentSnapshotPool
    // (currently not wired into EvolutionProtocolState at all)
    const stateWithPool = result.nextState as EvolutionProtocolState & {
      opponentSnapshotPool?: OpponentSnapshotPool;
    };
    expect(stateWithPool.opponentSnapshotPool).toBeDefined();
  });

  it('accumulates opponent snapshots across multiple generations', () => {
    // Arrange — racing state with a mock runner that reports done
    const racingState = createRacingStateWithDoneRunner();

    // Act — send request-race-step (first generation completes)
    const result = routeRacingWorkerProtocolMessage(
      { type: 'request-race-step', requestId: 'hof-2', stepsToAdvance: 1 },
      racingState,
    );

    // Assert — pool should have at least 1 snapshot after a generation
    // (currently pool is undefined, so snapshots.length is 0)
    const stateWithPool = result.nextState as EvolutionProtocolState & {
      opponentSnapshotPool?: OpponentSnapshotPool;
    };
    expect(
      (stateWithPool.opponentSnapshotPool?.snapshots.length ?? 0) >= 1,
    ).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Red tests — Core-to-race-pack snapshot type adapter
// ---------------------------------------------------------------------------

describe('Tier 6 core-to-race-pack opponent snapshot adapter', () => {
  it('exports a function to convert core OpponentSnapshot to race-pack shape', async () => {
    // Arrange — attempt to load the adapter function from the race-pack service
    let adapter: SnapshotAdapterModule | undefined;
    try {
      adapter = await loadSnapshotAdapter();
    } catch {
      adapter = undefined;
    }

    // Assert — the adapter function should exist
    // (currently not exported from simulation-worker.race-pack.service.ts)
    expect(typeof adapter?.convertCoreToRacePackSnapshot).toBe('function');
  });
});
