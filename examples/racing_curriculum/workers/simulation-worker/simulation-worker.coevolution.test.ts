/**
 * Red-phase contracts for the missing Team A/B coevolution container in
 * `simulation-worker.coevolution.service.ts` and the rolling opponent snapshot
 * store in `simulation-worker.opponent-snapshot.service.ts`.
 *
 * Contracts verified here:
 * - Team A and Team B are distinct independent population containers
 * - Team-level fitness = best (lowest) finishing position among the team's cars
 * - Opponent snapshot cannot be updated while evaluation is active (generation barrier)
 * - Opponent snapshot can be updated after evaluation ends at the configured boundary
 * - Opponent snapshot is not updated before the configured generation boundary
 *
 * All tests stay red until Step 04 implements the service boundaries.
 * Single-expect rule is enforced throughout.
 *
 * Role differentiation (queen, blocker, pacer) and radio semantics are NOT
 * prescribed here; they must remain emergent from evolution.
 *
 * TODO: NGE_TODO — When ModulatorBroadcaster and EpisodicSlot become available
 * (upstream Phase G), the coevolution container should support neuromodulation
 * broadcast across team members.
 *
 * TODO: NGE_TODO — Polyandric reproduction (`modeIsEvolvable`, upstream Phase E)
 * should extend the population container once that primitive exists.
 */

// ---------------------------------------------------------------------------
// Locally-defined interfaces
// ---------------------------------------------------------------------------

type CoevolutionConfig = {
  readonly populationSize: number;
  readonly rngSeed: number;
  readonly tier: number;
};

/**
 * Opaque handle for one team's independent population + species + fitness
 * state.  Step 04 wraps a real `Neat` instance behind this interface.
 */
type TeamPopulationContainer = {
  /** Opaque identity token — distinct between teamA and teamB. */
  readonly populationId: string;
};

type CoevolutionContainer = {
  readonly teamA: TeamPopulationContainer;
  readonly teamB: TeamPopulationContainer;
  /**
   * Returns the team's fitness score, defined as the best (lowest) finishing
   * position among the cars in `carFinishPositions`.
   *
   * @param teamId - 0 for Team A, 1 for Team B.
   * @param carFinishPositions - Finish positions for only that team's cars.
   *   Position 1 = first place (best).
   */
  resolveTeamFitness(
    teamId: 0 | 1,
    carFinishPositions: readonly number[],
  ): number;
};

interface CoevolutionService {
  createCoevolutionContainer(config: CoevolutionConfig): CoevolutionContainer;
}

// ---------------------------------------------------------------------------

type OpponentSnapshotConfig = {
  /** Update the frozen snapshot every this many generations. */
  readonly updateEveryNGenerations: number;
};

type OpponentSnapshotStore = {
  /** ID of the currently frozen snapshot; null before the first update. */
  readonly frozenSnapshotId: string | null;
  /** Returns true while an evaluation episode is active. */
  isEvaluationActive(): boolean;
  /** Marks the start of an evaluation episode (freezes the snapshot). */
  beginEvaluation(): void;
  /** Marks the end of an evaluation episode (releases the barrier). */
  endEvaluation(): void;
  /**
   * Attempts to update the frozen opponent snapshot.
   * Returns `true` when the update was applied; `false` when rejected
   * (evaluation active OR generation is before the configured boundary).
   */
  tryUpdateSnapshot(generation: number, payload: unknown): boolean;
};

interface OpponentSnapshotService {
  createOpponentSnapshotStore(
    config: OpponentSnapshotConfig,
  ): OpponentSnapshotStore;
}

// ---------------------------------------------------------------------------
// Module loaders
// ---------------------------------------------------------------------------

async function loadCoevolutionService(): Promise<CoevolutionService> {
  const modulePath = './simulation-worker.coevolution.service';
  return (await import(modulePath)) as CoevolutionService;
}

async function loadOpponentSnapshotService(): Promise<OpponentSnapshotService> {
  const modulePath = './simulation-worker.opponent-snapshot.service';
  return (await import(modulePath)) as OpponentSnapshotService;
}

// ---------------------------------------------------------------------------
// Red tests — coevolution container
// ---------------------------------------------------------------------------

describe('simulation worker coevolution container', () => {
  describe('independent Team A and Team B containers', () => {
    it('creates distinct population container objects for teamA and teamB', async () => {
      // Arrange
      const service = await loadCoevolutionService();

      // Act
      const container = service.createCoevolutionContainer({
        populationSize: 10,
        rngSeed: 1,
        tier: 1,
      });

      // Assert — teamA and teamB must not alias the same object
      expect(container.teamA).not.toBe(container.teamB);
    });

    it('assigns a distinct populationId to teamA versus teamB', async () => {
      // Arrange
      const service = await loadCoevolutionService();

      // Act
      const container = service.createCoevolutionContainer({
        populationSize: 10,
        rngSeed: 1,
        tier: 1,
      });

      // Assert — IDs must not collide so evolution state cannot cross-contaminate
      expect(container.teamA.populationId).not.toBe(
        container.teamB.populationId,
      );
    });
  });

  describe('team-level fitness resolution', () => {
    it('resolves team fitness as the best (lowest) finishing position among team cars', async () => {
      // Arrange
      const service = await loadCoevolutionService();
      const container = service.createCoevolutionContainer({
        populationSize: 10,
        rngSeed: 2,
        tier: 1,
      });

      // Act — cars at positions 3 and 7; best = 3
      const teamFitness = container.resolveTeamFitness(0, [3, 7]);

      // Assert
      expect(teamFitness).toBe(3);
    });

    it('does not average finish positions when resolving team fitness', async () => {
      // Arrange
      const service = await loadCoevolutionService();
      const container = service.createCoevolutionContainer({
        populationSize: 10,
        rngSeed: 3,
        tier: 1,
      });

      // Act — cars at positions 1 and 9; average = 5, best = 1
      const teamFitness = container.resolveTeamFitness(0, [1, 9]);

      // Assert — fitness must be 1 (best), not 5 (average)
      expect(teamFitness).toBe(1);
    });
  });
});

// ---------------------------------------------------------------------------
// Red tests — opponent snapshot store (generation barrier)
// ---------------------------------------------------------------------------

describe('simulation worker opponent snapshot store', () => {
  describe('generation barrier — no update during active evaluation', () => {
    it('rejects a snapshot update while evaluation is active', async () => {
      // Arrange
      const service = await loadOpponentSnapshotService();
      const store = service.createOpponentSnapshotStore({
        updateEveryNGenerations: 5,
      });
      store.beginEvaluation();

      // Act
      const updateApplied = store.tryUpdateSnapshot(5, { payload: 'blocked' });

      // Assert
      expect(updateApplied).toBe(false);
    });
  });

  describe('generation barrier — update accepted after evaluation ends at boundary', () => {
    it('accepts a snapshot update after evaluation ends at the configured generation boundary', async () => {
      // Arrange
      const service = await loadOpponentSnapshotService();
      const store = service.createOpponentSnapshotStore({
        updateEveryNGenerations: 5,
      });
      store.beginEvaluation();
      store.endEvaluation();

      // Act — generation 5 hits the boundary (5 % 5 === 0)
      const updateApplied = store.tryUpdateSnapshot(5, { payload: 'new' });

      // Assert
      expect(updateApplied).toBe(true);
    });
  });

  describe('generation barrier — no update before the configured boundary', () => {
    it('rejects a snapshot update when the generation has not yet reached the configured boundary', async () => {
      // Arrange
      const service = await loadOpponentSnapshotService();
      const store = service.createOpponentSnapshotStore({
        updateEveryNGenerations: 10,
      });

      // Act — generation 5 does not hit boundary (5 % 10 !== 0)
      const updateApplied = store.tryUpdateSnapshot(5, { payload: 'early' });

      // Assert
      expect(updateApplied).toBe(false);
    });
  });
});
