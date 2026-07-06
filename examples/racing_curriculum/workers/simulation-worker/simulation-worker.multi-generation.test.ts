/**
 * Red-phase contracts for the multi-generation evolution loop (Phase 7 Step 03).
 *
 * Contracts verified here:
 * - FSM handleRaceStep transitions to generation-ready when race reports done
 * - raceRunner is cleared from state after race-to-generation-ready transition
 * - Generation counter is non-zero in the generation-ready response after a race
 * - advanceTeamGeneration is called for both teams when a race completes
 * - carFitnessScores are populated with non-zero values after a race
 * - teamABestFitness is non-zero in the generation-ready response after a race
 *
 * All tests stay red until Step 04 fixes the 5 compounding FSM bugs:
 *   1. handleRaceStep returns nextState: currentState (stays in racing)
 *   2. buildGenerationReadyResponse hardcodes generation: 0
 *   3. advanceTeamGeneration is dead code (never called from protocol service)
 *   4. request-generation creates a fresh container every call
 *   5. tryUpdateSnapshot is dead code in opponent-snapshot service
 *
 * Single-expect rule is enforced throughout.
 *
 * Coverage closure tests (Step 04 Slice s1):
 * - extractCarFitnessScores with computeFitness path (lines 155-156)
 * - computeSharedEqualTeamFitness with empty team (line 125)
 */
import {
  routeRacingWorkerProtocolMessage,
  createInitialProtocolState,
  computeSharedEqualTeamFitness,
} from './simulation-worker.evolution.protocol.service';
import type {
  EvolutionProtocolState,
  GenerationReadyResponse,
} from './simulation-worker.evolution.types';
import type { CoevolutionContainer } from './simulation-worker.coevolution.service';

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
// Mock race episode runner with computeFitness — for fitness extraction path
// ---------------------------------------------------------------------------

interface MockRaceRunnerWithFitness extends MockRaceRunner {
  computeFitness(carIndex: number): number;
}

function createDoneMockRunnerWithFitness(): MockRaceRunnerWithFitness {
  return {
    tick: () => {},
    frame: { done: true },
    serializeVisualizationNetwork: () => new Float32Array([0]),
    computeFitness: (carIndex: number) => 100 + carIndex * 10,
  };
}

// ---------------------------------------------------------------------------
// Mock race episode runner with lap data — for finish-position fitness path
// ---------------------------------------------------------------------------

interface MockRaceRunnerWithLapData extends MockRaceRunner {
  lapCompleted: Uint8Array;
  lapTimeTicks: Uint32Array;
  frame: { done: true; progress01: Float32Array };
}

/**
 * Creates a 6-car mock runner where cars 0–2 completed laps and cars 3–5 did not.
 *
 * Finish order: car 2 (80 ticks) < car 1 (90 ticks) < car 0 (100 ticks),
 * then car 3 (0.9 progress) > car 4 (0.5) > car 5 (0.3).
 */
function createDoneMockRunnerWithLapData(): MockRaceRunnerWithLapData {
  return {
    tick: () => {},
    frame: {
      done: true,
      progress01: new Float32Array([1.0, 1.0, 1.0, 0.9, 0.5, 0.3]),
    },
    serializeVisualizationNetwork: () => new Float32Array([0]),
    lapCompleted: new Uint8Array([1, 1, 1, 0, 0, 0]),
    lapTimeTicks: new Uint32Array([100, 90, 80, 0, 0, 0]),
  };
}

// ---------------------------------------------------------------------------
// Helper: walk the FSM to a racing state, then swap in a done runner
// ---------------------------------------------------------------------------

/**
 * Drives the FSM from idle through init → request-generation → start-race,
 * then replaces the raceRunner with a mock that immediately reports done.
 *
 * This isolates the handleRaceStep transition logic without running full
 * physics. The coevolution container is real (created by request-generation
 * with tier 6 for the 6-car 3v3 layout).
 */
function createRacingStateWithDoneRunner(): EvolutionProtocolState {
  // Step 1: init with tier 6 (3v3 layout)
  const initResult = routeRacingWorkerProtocolMessage(
    { type: 'init', populationSize: 10, rngSeed: 42, tier: 6 },
    createInitialProtocolState(),
  );

  // Step 2: request-generation (creates coevolution container)
  const genResult = routeRacingWorkerProtocolMessage(
    { type: 'request-generation' },
    initResult.nextState,
  );

  // Step 3: start-race (creates race episode runner)
  const raceResult = routeRacingWorkerProtocolMessage(
    { type: 'start-race', tierConfig: null, opponentSnapshotId: 'test' },
    genResult.nextState,
  );

  // Step 4: Replace the race runner with a mock that reports done immediately
  return {
    ...raceResult.nextState,
    raceRunner: createDoneMockRunner(),
  };
}

// ---------------------------------------------------------------------------
// Red tests — FSM multi-generation loop transition (5 compounding bugs)
// ---------------------------------------------------------------------------

describe('Multi-generation FSM loop — racing to generation-ready transition', () => {
  it('transitions to generation-ready phase when race step reports done', () => {
    // Arrange — racing state with a mock runner that reports done
    const racingState = createRacingStateWithDoneRunner();

    // Act — send request-race-step with 1 step
    const result = routeRacingWorkerProtocolMessage(
      { type: 'request-race-step', requestId: 'r1', stepsToAdvance: 1 },
      racingState,
    );

    // Assert — should transition to generation-ready (currently stays racing)
    expect(result.nextState.phase).toBe('generation-ready');
  });

  it('clears raceRunner from state after transitioning to generation-ready', () => {
    // Arrange — racing state with a mock runner that reports done
    const racingState = createRacingStateWithDoneRunner();

    // Act — send request-race-step
    const result = routeRacingWorkerProtocolMessage(
      { type: 'request-race-step', requestId: 'r2', stepsToAdvance: 1 },
      racingState,
    );

    // Assert — raceRunner should be cleared (currently preserved)
    expect(result.nextState.raceRunner).toBeUndefined();
  });

  it('emits generation-ready response with non-zero generation after race completes', () => {
    // Arrange — racing state with a mock runner that reports done
    const racingState = createRacingStateWithDoneRunner();

    // Act — send request-race-step
    const result = routeRacingWorkerProtocolMessage(
      { type: 'request-race-step', requestId: 'r3', stepsToAdvance: 1 },
      racingState,
    );

    // Assert — response should be generation-ready with generation > 0
    // (currently emits race-step response, no generation field)
    const response = result.response as GenerationReadyResponse | undefined;
    expect((response?.generation ?? -1) > 0).toBe(true);
  });

  it('advances team A generation counter when race completes', () => {
    // Arrange — racing state with a real coevolution container
    const racingState = createRacingStateWithDoneRunner();

    // Act — send request-race-step (should trigger advanceTeamGeneration)
    const result = routeRacingWorkerProtocolMessage(
      { type: 'request-race-step', requestId: 'r4', stepsToAdvance: 1 },
      racingState,
    );

    // Assert — team A generation should be > 0 (advanceTeamGeneration is dead code)
    const container = result.nextState.coevolutionContainer as
      CoevolutionContainer | undefined;
    expect((container?.teamA.generation ?? -1) > 0).toBe(true);
  });

  it('advances team B generation counter when race completes', () => {
    // Arrange — racing state with a real coevolution container
    const racingState = createRacingStateWithDoneRunner();

    // Act — send request-race-step (should trigger advanceTeamGeneration)
    const result = routeRacingWorkerProtocolMessage(
      { type: 'request-race-step', requestId: 'r5', stepsToAdvance: 1 },
      racingState,
    );

    // Assert — team B generation should be > 0 (advanceTeamGeneration is dead code)
    const container = result.nextState.coevolutionContainer as
      CoevolutionContainer | undefined;
    expect((container?.teamB.generation ?? -1) > 0).toBe(true);
  });

  it('populates carFitnessScores with non-zero values after a race', () => {
    // Arrange — racing state with a mock runner that reports done
    const racingState = createRacingStateWithDoneRunner();

    // Act — send request-race-step
    const result = routeRacingWorkerProtocolMessage(
      { type: 'request-race-step', requestId: 'r6', stepsToAdvance: 1 },
      racingState,
    );

    // Assert — carFitnessScores should have at least one non-zero value
    // (currently hardcoded to all zeros in buildGenerationReadyResponse)
    const response = result.response as GenerationReadyResponse | undefined;
    const fitnessScores = response?.carFitnessScores ?? [];
    expect(fitnessScores.some((score) => score !== 0)).toBe(true);
  });

  it('records non-zero team A best fitness after a race', () => {
    // Arrange — racing state with a mock runner that reports done
    const racingState = createRacingStateWithDoneRunner();

    // Act — send request-race-step
    const result = routeRacingWorkerProtocolMessage(
      { type: 'request-race-step', requestId: 'r7', stepsToAdvance: 1 },
      racingState,
    );

    // Assert — teamABestFitness should be non-zero
    // (currently hardcoded to 0 via computeSharedEqualTeamFitness on all-zero scores)
    const response = result.response as GenerationReadyResponse | undefined;
    expect(response?.teamABestFitness ?? 0).not.toBe(0);
  });

  // -------------------------------------------------------------------------
  // Coverage closure — extractCarFitnessScores computeFitness path (lines 155-156)
  // -------------------------------------------------------------------------

  it('extracts per-car fitness from runner computeFitness when available', () => {
    // Arrange — racing state with a mock runner that exposes computeFitness
    const racingState = {
      ...createRacingStateWithDoneRunner(),
      raceRunner: createDoneMockRunnerWithFitness(),
    };

    // Act — send request-race-step to trigger transitionToGenerationReady
    const result = routeRacingWorkerProtocolMessage(
      { type: 'request-race-step', requestId: 'r8', stepsToAdvance: 1 },
      racingState,
    );

    // Assert — fitness scores come from computeFitness, not the fallback
    const response = result.response as GenerationReadyResponse | undefined;
    expect(response?.carFitnessScores).toEqual([100, 110, 120, 130, 140, 150]);
  });

  // -------------------------------------------------------------------------
  // Coverage closure — computeSharedEqualTeamFitness empty team guard (line 125)
  // -------------------------------------------------------------------------

  it('returns 0 when the team has no members', () => {
    // All cars on team 1 (red), so team 0 (blue) has no members
    const result = computeSharedEqualTeamFitness([10, 20, 30], [1, 1, 1], 0);
    expect(result).toBe(0);
  });

  // -------------------------------------------------------------------------
  // Coverage closure — tryExtractFinishPositions lap-data path
  // -------------------------------------------------------------------------

  it('derives per-car fitness from real finish positions when lap data is available', () => {
    // Arrange — racing state with a mock runner that exposes lap data
    const racingState = {
      ...createRacingStateWithDoneRunner(),
      raceRunner: createDoneMockRunnerWithLapData(),
    };

    // Act — send request-race-step to trigger transitionToGenerationReady
    const result = routeRacingWorkerProtocolMessage(
      { type: 'request-race-step', requestId: 'r9', stepsToAdvance: 1 },
      racingState,
    );

    // Assert — fitness scores come from finish positions, not the fallback
    // Car 2 (80 ticks) → rank 0 → 1600; car 1 (90) → rank 1 → 1500;
    // car 0 (100) → rank 2 → 1400; car 3 (0.9 progress) → rank 3 → 300;
    // car 4 (0.5) → rank 4 → 200; car 5 (0.3) → rank 5 → 100.
    const response = result.response as GenerationReadyResponse | undefined;
    expect(response?.carFitnessScores).toEqual([
      1400, 1500, 1600, 300, 200, 100,
    ]);
  });
});
