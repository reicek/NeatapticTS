/**
 * Red-phase contracts for the missing worker-authoritative evolution protocol
 * FSM in `simulation-worker.evolution.protocol.service.ts`.
 *
 * Lifecycle: idle → initialised → generation-ready → racing → stopped
 *
 * All tests stay red until Step 04 implements the service boundary.
 * Single-expect rule is enforced throughout.
 */

// ---------------------------------------------------------------------------
// Locally-defined interface — keeps TypeScript happy without implementing the
// feature. Dynamic import path is assigned to a runtime string so ts-jest
// cannot statically resolve it (same pattern as simulation-worker.tier3.test.ts).
// ---------------------------------------------------------------------------

/** Phase labels for the worker-authoritative evolution FSM. */
type RacingWorkerPhase =
  'idle' | 'initialised' | 'generation-ready' | 'racing' | 'stopped';

type EvolutionProtocolState = {
  readonly phase: RacingWorkerPhase;
};

type RacingWorkerInboundMessage =
  | { type: 'init'; populationSize: number; rngSeed: number; tier: number }
  | { type: 'request-generation' }
  | { type: 'start-race'; tierConfig: unknown; opponentSnapshotId: string }
  | { type: 'request-race-step'; requestId: string; stepsToAdvance: number }
  | { type: 'stop' };

type GenerationReadyResponse = {
  readonly type: 'generation-ready';
  readonly generation: number;
  readonly teamABestFitness: number;
  readonly teamBBestFitness: number;
  readonly bestNetworkPayload?: unknown;
  /** Optional zero-copy transfer list for the generation payload. */
  readonly transferList?: readonly ArrayBuffer[];
};

type EvolutionProtocolRouteResult = {
  readonly nextState: EvolutionProtocolState;
  readonly response?: GenerationReadyResponse | unknown;
  /** Populated when the message is rejected in the current phase. */
  readonly error?: string;
};

interface EvolutionProtocolService {
  /** Returns the canonical starting state: phase = 'idle'. */
  createInitialProtocolState(): EvolutionProtocolState;
  /** Routes one inbound message through the FSM and returns the next state + optional response. */
  routeRacingWorkerProtocolMessage(
    message: RacingWorkerInboundMessage,
    state: EvolutionProtocolState,
  ): EvolutionProtocolRouteResult;
}

// ---------------------------------------------------------------------------
// Module loader
// ---------------------------------------------------------------------------

async function loadEvolutionProtocolService(): Promise<EvolutionProtocolService> {
  const modulePath = './simulation-worker.evolution.protocol.service';
  return (await import(modulePath)) as EvolutionProtocolService;
}

// ---------------------------------------------------------------------------
// Red tests
// ---------------------------------------------------------------------------

describe('simulation worker evolution protocol FSM', () => {
  describe('request-generation before init', () => {
    it('rejects request-generation message when the worker is in idle phase', async () => {
      // Arrange
      const service = await loadEvolutionProtocolService();
      const idleState = service.createInitialProtocolState();

      // Act
      const result = service.routeRacingWorkerProtocolMessage(
        { type: 'request-generation' },
        idleState,
      );

      // Assert
      expect(result.error).toBeDefined();
    });
  });

  describe('request-generation in initialised phase', () => {
    it('returns a generation-ready response when request-generation is routed', async () => {
      // Arrange
      const service = await loadEvolutionProtocolService();
      const initialisedState: EvolutionProtocolState = { phase: 'initialised' };

      // Act
      const result = service.routeRacingWorkerProtocolMessage(
        { type: 'request-generation' },
        initialisedState,
      );
      const response = result.response as GenerationReadyResponse | undefined;

      // Assert
      expect(response?.type).toBe('generation-ready');
    });

    it('includes a non-empty transfer list in the generation-ready response', async () => {
      // Arrange
      const service = await loadEvolutionProtocolService();
      const initialisedState: EvolutionProtocolState = { phase: 'initialised' };

      // Act
      const result = service.routeRacingWorkerProtocolMessage(
        { type: 'request-generation' },
        initialisedState,
      );
      const response = result.response as GenerationReadyResponse | undefined;

      // Assert
      expect(response?.transferList?.length ?? 0).toBeGreaterThan(0);
    });
  });

  describe('start-race before any generation exists', () => {
    it('rejects start-race message when the worker is in initialised phase', async () => {
      // Arrange
      const service = await loadEvolutionProtocolService();
      const initialisedState: EvolutionProtocolState = { phase: 'initialised' };

      // Act
      const result = service.routeRacingWorkerProtocolMessage(
        { type: 'start-race', tierConfig: {}, opponentSnapshotId: 'snap-0' },
        initialisedState,
      );

      // Assert
      expect(result.error).toBeDefined();
    });
  });

  describe('request-race-step before start-race', () => {
    it('rejects request-race-step message when the worker is in generation-ready phase', async () => {
      // Arrange
      const service = await loadEvolutionProtocolService();
      const generationReadyState: EvolutionProtocolState = {
        phase: 'generation-ready',
      };

      // Act
      const result = service.routeRacingWorkerProtocolMessage(
        { type: 'request-race-step', requestId: 'r-1', stepsToAdvance: 1 },
        generationReadyState,
      );

      // Assert
      expect(result.error).toBeDefined();
    });
  });

  describe('stop message routing', () => {
    it('transitions to stopped phase from idle regardless of prior state', async () => {
      // Arrange
      const service = await loadEvolutionProtocolService();
      const idleState = service.createInitialProtocolState();

      // Act
      const result = service.routeRacingWorkerProtocolMessage(
        { type: 'stop' },
        idleState,
      );

      // Assert
      expect(result.nextState.phase).toBe('stopped');
    });
  });
});
