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
// Type imports from the coevolution service
// ---------------------------------------------------------------------------

import type {
  CoevolutionConfig,
  CoevolutionContainer,
} from './simulation-worker.coevolution.service';

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

  describe('independent team generation advancement', () => {
    it('advances teamA generation when asked', async () => {
      // Arrange
      const service = await loadCoevolutionService();
      const container = service.createCoevolutionContainer({
        populationSize: 6,
        rngSeed: 7,
        tier: 1,
      });

      // Act
      container.advanceTeamGeneration('team-a');

      // Assert — teamA must have moved forward exactly one generation
      expect(container.teamA.generation).toBe(1);
    });

    it('does not advance teamB generation when advancing teamA', async () => {
      // Arrange
      const service = await loadCoevolutionService();
      const container = service.createCoevolutionContainer({
        populationSize: 6,
        rngSeed: 7,
        tier: 1,
      });

      // Act
      container.advanceTeamGeneration('team-a');

      // Assert — teamB must remain at generation zero
      expect(container.teamB.generation).toBe(0);
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
// Tier 3 red tests — four-distinct-genome coevolution
// ---------------------------------------------------------------------------

describe('Tier 3 four-car genome layout', () => {
  it('produces four distinct genomes for a tier-3 container', async () => {
    // Arrange
    const service = await loadCoevolutionService();

    // Act
    const container = service.createCoevolutionContainer({
      populationSize: 10,
      rngSeed: 42,
      tier: 3,
    });
    const genomes = container.getCarGenomes();

    // Assert — must return 4 genomes (2 per team), not 2
    expect(genomes.length).toBe(4);
  });

  it('assigns team layout [0, 0, 1, 1] for four cars', async () => {
    // Arrange
    const service = await loadCoevolutionService();

    // Act
    const container = service.createCoevolutionContainer({
      populationSize: 10,
      rngSeed: 42,
      tier: 3,
    });
    const genomes = container.getCarGenomes();
    const teamLayout = genomes.map((genome) => genome.teamId);

    // Assert — cars 0-1 = Team A, cars 2-3 = Team B
    expect(teamLayout).toEqual([0, 0, 1, 1]);
  });

  it('produces four distinct genome instances with no shared references', async () => {
    // Arrange
    const service = await loadCoevolutionService();

    // Act
    const container = service.createCoevolutionContainer({
      populationSize: 10,
      rngSeed: 42,
      tier: 3,
    });
    const genomes = container.getCarGenomes();

    // Assert — each genome must be a distinct object
    const allDistinct =
      genomes[0] !== genomes[1] &&
      genomes[0] !== genomes[2] &&
      genomes[0] !== genomes[3] &&
      genomes[1] !== genomes[2] &&
      genomes[1] !== genomes[3] &&
      genomes[2] !== genomes[3];

    expect(allDistinct).toBe(true);
  });

  it('assigns carIndex 0 through 3 to the four genomes', async () => {
    // Arrange
    const service = await loadCoevolutionService();

    // Act
    const container = service.createCoevolutionContainer({
      populationSize: 10,
      rngSeed: 42,
      tier: 3,
    });
    const genomes = container.getCarGenomes();
    const carIndices = genomes.map((genome) => genome.carIndex);

    // Assert
    expect(carIndices).toEqual([0, 1, 2, 3]);
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

// ---------------------------------------------------------------------------
// Tier 4 red tests — tire-aware genome input size
// ---------------------------------------------------------------------------

describe('Tier 4 tire-aware genome input size', () => {
  it('creates a 103-input controller network for the first tier-4 car genome', async () => {
    // Arrange
    const service = await loadCoevolutionService();

    // Act
    const container = service.createCoevolutionContainer({
      populationSize: 10,
      rngSeed: 42,
      tier: 4,
    });
    const firstGenome = container.getCarGenome(0);

    // Assert — Tier 4 must use 103 inputs (91 Tier 3 + 4 tire + 8 pit/strategy channels), not 91
    expect(firstGenome.getNetwork().input).toBe(103);
  });

  it('creates 103-input controller networks for all four tier-4 car genomes', async () => {
    // Arrange
    const service = await loadCoevolutionService();

    // Act
    const container = service.createCoevolutionContainer({
      populationSize: 10,
      rngSeed: 42,
      tier: 4,
    });
    const genomes = container.getCarGenomes();
    const inputSizes = genomes.map((genome) => genome.getNetwork().input);

    // Assert — all four Tier 4 genomes must have 103 inputs
    expect(inputSizes).toEqual([103, 103, 103, 103]);
  });
});

// ---------------------------------------------------------------------------
// Tier 5 red tests — six-car 3v3 coevolution container
// ---------------------------------------------------------------------------

describe('Tier 5 six-car genome layout', () => {
  it('produces six distinct genomes for a tier-5 container', async () => {
    // Arrange
    const service = await loadCoevolutionService();

    // Act
    const container = service.createCoevolutionContainer({
      populationSize: 10,
      rngSeed: 42,
      tier: 5,
    });
    const genomes = container.getCarGenomes();

    // Assert — must return 6 genomes (3 per team), not 4
    expect(genomes.length).toBe(6);
  });

  it('assigns team layout [0, 0, 0, 1, 1, 1] for six cars', async () => {
    // Arrange
    const service = await loadCoevolutionService();

    // Act
    const container = service.createCoevolutionContainer({
      populationSize: 10,
      rngSeed: 42,
      tier: 5,
    });
    const genomes = container.getCarGenomes();
    const teamLayout = genomes.map((genome) => genome.teamId);

    // Assert — cars 0-2 = Team A, cars 3-5 = Team B
    expect(teamLayout).toEqual([0, 0, 0, 1, 1, 1]);
  });

  it('assigns carIndex 0 through 5 to the six genomes', async () => {
    // Arrange
    const service = await loadCoevolutionService();

    // Act
    const container = service.createCoevolutionContainer({
      populationSize: 10,
      rngSeed: 42,
      tier: 5,
    });
    const genomes = container.getCarGenomes();
    const carIndices = genomes.map((genome) => genome.carIndex);

    // Assert
    expect(carIndices).toEqual([0, 1, 2, 3, 4, 5]);
  });

  it('produces a defined genome at carIndex 5', async () => {
    // Arrange
    const service = await loadCoevolutionService();

    // Act
    const container = service.createCoevolutionContainer({
      populationSize: 10,
      rngSeed: 42,
      tier: 5,
    });
    const sixthGenome = container.getCarGenome(5);

    // Assert — the sixth genome must exist (currently only 4 are allocated)
    expect(sixthGenome).toBeDefined();
  });
});
