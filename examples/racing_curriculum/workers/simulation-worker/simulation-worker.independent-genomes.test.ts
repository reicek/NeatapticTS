/**
 * Red-phase contracts for per-car independent genomes in the racing curriculum
 * worker.
 *
 * The racing curriculum demo has multiple independent NGE agent cars. Each car
 * is a fully independent network with continuous evolution. The worker
 * evaluation path must provide one distinct genome per car. The coevolution
 * container must not share genomes across cars.
 *
 * Decision: All cars coevolve independently. Each car gets its own distinct
 * genome. Only the blue team #1 (carIndex 0) network is copied back to the
 * browser for visualization. Other cars' networks stay in the worker.
 *
 * Contracts verified here:
 * - The coevolution container provides a different genome for each car index
 * - The evolution protocol returns per-car/per-team payloads, not a single
 *   shared payload
 * - Modifying car 0's genome does NOT affect car 1's genome
 * - The protocol supports copying back only car 0's network for visualization
 * - The coevolution container has a real implementation, not a stub
 *
 * All tests stay red until the implementation phase adds per-car genome
 * support to the coevolution container and evolution protocol.
 * Single-expect rule is enforced throughout.
 */

// ---------------------------------------------------------------------------
// Locally-defined interfaces — expected API surface for per-car genomes.
// ---------------------------------------------------------------------------

type CoevolutionConfig = {
  readonly populationSize: number;
  readonly rngSeed: number;
  readonly tier: number;
};

type TeamPopulationContainer = {
  readonly populationId: string;
  generation: number;
};

/**
 * Handle for one car's independent genome.
 *
 * Each car genome is a fully independent network with its own evolution state.
 * Mutating one car's genome must not affect any other car's genome.
 */
type CarGenome = {
  /** Car index within the race pack (0-based). */
  readonly carIndex: number;
  /** Team id: 0 for Team A (blue), 1 for Team B (red). */
  readonly teamId: 0 | 1;
  /** Population id of the team this car belongs to. */
  readonly populationId: string;
  /** Runs inference and returns the controller output vector. */
  activate(inputs: number[]): number[];
  /** Mutates this genome in place; must not affect other cars' genomes. */
  mutate(): void;
};

type CoevolutionContainer = {
  readonly teamA: TeamPopulationContainer;
  readonly teamB: TeamPopulationContainer;
  resolveTeamFitness(
    teamId: 0 | 1,
    carFinishPositions: readonly number[],
  ): number;
  advanceTeamGeneration(teamId: 'team-a' | 'team-b'): void;
  /**
   * Returns the distinct genome for the requested car index.
   * Each car gets its own independent genome — no sharing across cars.
   */
  getCarGenome(carIndex: number): CarGenome;
  /**
   * Returns all car genomes as an array (one per car).
   * Each entry must be a distinct genome object.
   */
  getCarGenomes(): readonly CarGenome[];
};

interface CoevolutionService {
  createCoevolutionContainer(config: CoevolutionConfig): CoevolutionContainer;
}

// ---------------------------------------------------------------------------
// Evolution protocol interfaces — expected per-car payload surface.
// ---------------------------------------------------------------------------

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

/**
 * Generation-ready response with per-car payloads.
 *
 * The response must include per-car network payloads (one per car), not a
 * single shared payload. Only car 0's network is copied back for browser
 * visualization via `visualizationPayload`.
 */
type GenerationReadyResponse = {
  readonly type: 'generation-ready';
  readonly generation: number;
  readonly teamABestFitness: number;
  readonly teamBBestFitness: number;
  readonly bestNetworkPayload?: unknown;
  readonly transferList?: readonly ArrayBuffer[];
  /** Per-car network payloads — one entry per car, not a single shared payload. */
  readonly carNetworkPayloads?: readonly unknown[];
  /** Per-car fitness scores — one entry per car. */
  readonly carFitnessScores?: readonly number[];
  /**
   * Car index whose network is copied back for browser visualization.
   * Must be 0 (blue team #1) per the independent-genome architecture decision.
   */
  readonly visualizationCarIndex?: number;
  /**
   * Serialized network payload for the visualization car (car 0).
   * Must be a real network payload, not a placeholder Float32Array([0]).
   */
  readonly visualizationPayload?: unknown;
};

type EvolutionProtocolRouteResult = {
  readonly nextState: EvolutionProtocolState;
  readonly response?: unknown;
  readonly error?: string;
};

interface EvolutionProtocolService {
  createInitialProtocolState(): EvolutionProtocolState;
  routeRacingWorkerProtocolMessage(
    message: RacingWorkerInboundMessage,
    state: EvolutionProtocolState,
  ): EvolutionProtocolRouteResult;
}

// ---------------------------------------------------------------------------
// Module loaders
// ---------------------------------------------------------------------------

async function loadCoevolutionService(): Promise<CoevolutionService> {
  const modulePath = './simulation-worker.coevolution.service';
  return (await import(modulePath)) as CoevolutionService;
}

async function loadEvolutionProtocolService(): Promise<EvolutionProtocolService> {
  const modulePath = './simulation-worker.evolution.protocol.service';
  return (await import(modulePath)) as EvolutionProtocolService;
}

// ---------------------------------------------------------------------------
// Red tests — distinct genome per car
// ---------------------------------------------------------------------------

describe('simulation worker coevolution container per-car genomes', () => {
  describe('distinct genome per car', () => {
    it('returns a genome object for car index 0', async () => {
      // Arrange
      const service = await loadCoevolutionService();
      const container = service.createCoevolutionContainer({
        populationSize: 10,
        rngSeed: 1,
        tier: 1,
      });

      // Act
      const genome = container.getCarGenome(0);

      // Assert — must return a defined genome object, not undefined
      expect(genome).toBeDefined();
    });

    it('returns a genome object for car index 1', async () => {
      // Arrange
      const service = await loadCoevolutionService();
      const container = service.createCoevolutionContainer({
        populationSize: 10,
        rngSeed: 1,
        tier: 1,
      });

      // Act
      const genome = container.getCarGenome(1);

      // Assert — must return a defined genome object, not undefined
      expect(genome).toBeDefined();
    });

    it('returns distinct genome objects for car 0 and car 1', async () => {
      // Arrange
      const service = await loadCoevolutionService();
      const container = service.createCoevolutionContainer({
        populationSize: 10,
        rngSeed: 1,
        tier: 1,
      });

      // Act
      const genome0 = container.getCarGenome(0);
      const genome1 = container.getCarGenome(1);

      // Assert — genomes must not be the same object reference
      expect(genome0).not.toBe(genome1);
    });

    it('returns genomes with the correct carIndex field', async () => {
      // Arrange
      const service = await loadCoevolutionService();
      const container = service.createCoevolutionContainer({
        populationSize: 10,
        rngSeed: 1,
        tier: 1,
      });

      // Act
      const genome0 = container.getCarGenome(0);
      const genome1 = container.getCarGenome(1);

      // Assert — each genome must carry its own carIndex
      expect({ car0: genome0.carIndex, car1: genome1.carIndex }).toEqual({
        car0: 0,
        car1: 1,
      });
    });

    it('returns all car genomes from getCarGenomes with one entry per car', async () => {
      // Arrange
      const service = await loadCoevolutionService();
      const container = service.createCoevolutionContainer({
        populationSize: 10,
        rngSeed: 1,
        tier: 1,
      });

      // Act
      const genomes = container.getCarGenomes();

      // Assert — must return an array with at least 2 entries (one per car)
      expect(genomes.length).toBeGreaterThanOrEqual(2);
    });
  });

  // -------------------------------------------------------------------------
  // Red tests — no shared genome state
  // -------------------------------------------------------------------------

  describe('genome independence — no shared state between cars', () => {
    it('does not affect car 1 genome when car 0 genome is mutated', async () => {
      // Arrange
      const service = await loadCoevolutionService();
      const container = service.createCoevolutionContainer({
        populationSize: 10,
        rngSeed: 1,
        tier: 1,
      });
      const genome0 = container.getCarGenome(0);
      const genome1 = container.getCarGenome(1);
      const genome1OutputBefore = genome1.activate([0, 0, 0, 0]);

      // Act — mutate car 0's genome
      genome0.mutate();
      const genome1OutputAfter = genome1.activate([0, 0, 0, 0]);

      // Assert — car 1's output must not change after car 0's mutation
      expect(genome1OutputAfter).toEqual(genome1OutputBefore);
    });

    it('produces different activation outputs for car 0 and car 1 before any mutation', async () => {
      // Arrange
      const service = await loadCoevolutionService();
      const container = service.createCoevolutionContainer({
        populationSize: 10,
        rngSeed: 1,
        tier: 1,
      });

      // Act
      const genome0 = container.getCarGenome(0);
      const genome1 = container.getCarGenome(1);
      const inputs = [0.5, 0.5, 0.5, 0.5];
      const output0 = genome0.activate(inputs);
      const output1 = genome1.activate(inputs);

      // Assert — independent genomes should produce different outputs
      expect(output0).not.toEqual(output1);
    });
  });

  // -------------------------------------------------------------------------
  // Red tests — coevolution container is not a placeholder
  // -------------------------------------------------------------------------

  describe('coevolution container has a real implementation', () => {
    it('getCarGenome is a function on the container', async () => {
      // Arrange
      const service = await loadCoevolutionService();
      const container = service.createCoevolutionContainer({
        populationSize: 10,
        rngSeed: 1,
        tier: 1,
      });

      // Assert — the container must expose getCarGenome as a real function
      expect(typeof container.getCarGenome).toBe('function');
    });

    it('getCarGenomes is a function on the container', async () => {
      // Arrange
      const service = await loadCoevolutionService();
      const container = service.createCoevolutionContainer({
        populationSize: 10,
        rngSeed: 1,
        tier: 1,
      });

      // Assert — the container must expose getCarGenomes as a real function
      expect(typeof container.getCarGenomes).toBe('function');
    });

    it('car genome has an activate function for controller inference', async () => {
      // Arrange
      const service = await loadCoevolutionService();
      const container = service.createCoevolutionContainer({
        populationSize: 10,
        rngSeed: 1,
        tier: 1,
      });

      // Act
      const genome = container.getCarGenome(0);

      // Assert — genome must be a real network handle with an activate method
      expect(typeof genome.activate).toBe('function');
    });

    it('car genome has a mutate function for independent evolution', async () => {
      // Arrange
      const service = await loadCoevolutionService();
      const container = service.createCoevolutionContainer({
        populationSize: 10,
        rngSeed: 1,
        tier: 1,
      });

      // Act
      const genome = container.getCarGenome(0);

      // Assert — genome must support mutation for independent evolution
      expect(typeof genome.mutate).toBe('function');
    });
  });
});

// ---------------------------------------------------------------------------
// Red tests — per-car evolution payloads
// ---------------------------------------------------------------------------

describe('simulation worker evolution protocol per-car payloads', () => {
  describe('generation-ready response includes per-car payloads', () => {
    it('includes a carNetworkPayloads array in the generation-ready response', async () => {
      // Arrange
      const service = await loadEvolutionProtocolService();
      const initialisedState: EvolutionProtocolState = { phase: 'initialised' };

      // Act
      const result = service.routeRacingWorkerProtocolMessage(
        { type: 'request-generation' },
        initialisedState,
      );
      const response = result.response as GenerationReadyResponse | undefined;

      // Assert — must have per-car network payloads, not a single shared payload
      expect(Array.isArray(response?.carNetworkPayloads)).toBe(true);
    });

    it('includes at least two per-car network payloads (one per car)', async () => {
      // Arrange
      const service = await loadEvolutionProtocolService();
      const initialisedState: EvolutionProtocolState = { phase: 'initialised' };

      // Act
      const result = service.routeRacingWorkerProtocolMessage(
        { type: 'request-generation' },
        initialisedState,
      );
      const response = result.response as GenerationReadyResponse | undefined;

      // Assert — must have at least 2 entries (one per car)
      expect(response?.carNetworkPayloads?.length ?? 0).toBeGreaterThanOrEqual(
        2,
      );
    });

    it('includes per-car fitness scores in the generation-ready response', async () => {
      // Arrange
      const service = await loadEvolutionProtocolService();
      const initialisedState: EvolutionProtocolState = { phase: 'initialised' };

      // Act
      const result = service.routeRacingWorkerProtocolMessage(
        { type: 'request-generation' },
        initialisedState,
      );
      const response = result.response as GenerationReadyResponse | undefined;

      // Assert — must have per-car fitness scores array
      expect(Array.isArray(response?.carFitnessScores)).toBe(true);
    });
  });

  // -------------------------------------------------------------------------
  // Red tests — blue team #1 visualization copy
  // -------------------------------------------------------------------------

  describe('blue team #1 (car 0) visualization copy', () => {
    it('includes a visualizationCarIndex field set to 0 in the generation-ready response', async () => {
      // Arrange
      const service = await loadEvolutionProtocolService();
      const initialisedState: EvolutionProtocolState = { phase: 'initialised' };

      // Act
      const result = service.routeRacingWorkerProtocolMessage(
        { type: 'request-generation' },
        initialisedState,
      );
      const response = result.response as GenerationReadyResponse | undefined;

      // Assert — visualization must target car 0 (blue team #1)
      expect(response?.visualizationCarIndex).toBe(0);
    });

    it('includes a non-undefined visualizationPayload for the browser copy', async () => {
      // Arrange
      const service = await loadEvolutionProtocolService();
      const initialisedState: EvolutionProtocolState = { phase: 'initialised' };

      // Act
      const result = service.routeRacingWorkerProtocolMessage(
        { type: 'request-generation' },
        initialisedState,
      );
      const response = result.response as GenerationReadyResponse | undefined;
      const payload = response?.visualizationPayload;

      // Assert — visualization payload must be defined (a real network payload)
      expect(payload).toBeDefined();
    });
  });
});
