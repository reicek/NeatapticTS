/**
 * Rolling opponent snapshot store for the racing curriculum benchmark.
 *
 * The store enforces two guards before applying a snapshot update:
 * 1. Generation barrier — updates are rejected while an evaluation episode is
 *    active.  `beginEvaluation()` sets the barrier; `endEvaluation()` releases
 *    it.
 * 2. Generation boundary — updates are only applied at multiples of
 *    `updateEveryNGenerations` (i.e. when `generation % updateEveryNGenerations === 0`).
 *
 * The hall-of-fame / recent-pool sampling mirrors the common coevolution
 * stabilization technique of evaluating candidates against both strong historical
 * opponents and the current adversary, rather than against a single moving
 * target.  See [Coevolution (Wikipedia)](https://en.wikipedia.org/wiki/Coevolution)
 * for background.
 *
 * Extension point:
 * - Extend the snapshot payload format to include episodic context and hard
 *   task-switch state when `EpisodicSlot` and `GatingRouter` primitives are
 *   available.
 */

/** Snapshot update cadence configuration. */
export type OpponentSnapshotConfig = {
  /** Apply a snapshot update every this many generations. */
  readonly updateEveryNGenerations: number;
};

/**
 * One sampled opponent snapshot together with its source pool label.
 *
 * The label lets the caller distinguish historical hall-of-fame opponents from
 * recent opponents when building a mixed evaluation pool.
 */
export type SnapshotSample = {
  /** Stable identifier from the frozen opponent snapshot pool. */
  readonly snapshotId: string;
  /** Pool the snapshot was drawn from. */
  readonly source: 'hall-of-fame' | 'recent';
};

/**
 * Rolling opponent snapshot store with generation barrier enforcement.
 *
 * The store freezes a snapshot between evaluation episodes and only allows an
 * update when the configured generation boundary is crossed.
 */
export type OpponentSnapshotStore = {
  /** ID of the currently frozen snapshot; `null` before the first update. */
  readonly frozenSnapshotId: string | null;
  /** Payload of the currently frozen snapshot; `null` before the first update. */
  readonly frozenPayload: unknown;
  /** Returns `true` while an evaluation episode is active. */
  isEvaluationActive(): boolean;
  /** Marks the start of an evaluation episode, freezing the current snapshot. */
  beginEvaluation(): void;
  /** Marks the end of an evaluation episode, releasing the barrier. */
  endEvaluation(): void;
  /**
   * Attempts to replace the frozen opponent snapshot.
   *
   * Returns `true` when both guards pass and the snapshot is replaced.
   * Returns `false` when:
   * - An evaluation episode is currently active (generation barrier), or
   * - The generation has not yet reached the configured boundary.
   *
   * @param generation - Current generation index (1-based).
   * @param payload - Snapshot data to freeze.
   */
  tryUpdateSnapshot(generation: number, payload: unknown): boolean;
  /**
   * Samples opponent snapshot IDs from the hall-of-fame and recent pools.
   *
   * Guarantees at least one hall-of-fame sample when both pools are non-empty.
   *
   * @param options - Hall-of-fame IDs, recent IDs, and desired sample count.
   * @returns Ordered sample list drawn only from the provided pools.
   */
  sampleOpponentSnapshots(options: {
    hallOfFameIds: readonly string[];
    recentIds: readonly string[];
    sampleCount: number;
  }): readonly SnapshotSample[];
};

/** Monotonic counter for generating unique snapshot IDs. */
let snapshotSerialNumber = 0;

/**
 * Creates a rolling opponent snapshot store that respects the generation
 * barrier and the configured update boundary.
 *
 * @param config - Snapshot update policy.
 * @returns Opponent snapshot store with barrier-enforced update semantics.
 *
 * @example
 * ```ts
 * const store = createOpponentSnapshotStore({ updateEveryNGenerations: 5 });
 * store.beginEvaluation();
 * store.tryUpdateSnapshot(5, payload); // → false (barrier active)
 * store.endEvaluation();
 * store.tryUpdateSnapshot(5, payload); // → true  (barrier cleared, at boundary)
 * store.tryUpdateSnapshot(7, payload); // → false (not at boundary)
 * ```
 */
export function createOpponentSnapshotStore(
  config: OpponentSnapshotConfig,
): OpponentSnapshotStore {
  let evaluationActive = false;
  let frozenSnapshotId: string | null = null;
  let frozenPayload: unknown = null;

  return {
    get frozenSnapshotId() {
      return frozenSnapshotId;
    },
    get frozenPayload() {
      return frozenPayload;
    },
    isEvaluationActive,
    beginEvaluation,
    endEvaluation,
    tryUpdateSnapshot,
    sampleOpponentSnapshots,
  };

  /** Returns whether an evaluation episode is currently active. */
  function isEvaluationActive(): boolean {
    return evaluationActive;
  }

  /** Freezes the snapshot barrier at the start of an evaluation episode. */
  function beginEvaluation(): void {
    evaluationActive = true;
  }

  /** Releases the snapshot barrier at the end of an evaluation episode. */
  function endEvaluation(): void {
    evaluationActive = false;
  }

  /**
   * Applies the snapshot update when both guards pass.
   *
   * Guard 1: generation barrier must not be active.
   * Guard 2: `generation % updateEveryNGenerations === 0`.
   */
  function tryUpdateSnapshot(generation: number, payload: unknown): boolean {
    // Step 1: Reject if the generation barrier is active.
    if (evaluationActive) {
      return false;
    }

    // Step 2: Reject if the generation has not reached the configured boundary.
    if (generation % config.updateEveryNGenerations !== 0) {
      return false;
    }

    // Step 3: Both guards passed — replace the frozen snapshot.
    frozenSnapshotId = `snapshot-${++snapshotSerialNumber}-gen${generation}`;
    frozenPayload = payload;
    return true;
  }

  /**
   * Samples opponent snapshots from the provided hall-of-fame and recent pools.
   *
   * The returned list always contains `sampleCount` entries. When both pools are
   * non-empty the first entry is guaranteed to come from the hall-of-fame pool
   * so the racing episode has at least one high-quality reference opponent.
   *
   * @param options - Hall-of-fame IDs, recent IDs, and desired sample count.
   * @returns Ordered sample list containing only IDs from the provided pools.
   */
  function sampleOpponentSnapshots({
    hallOfFameIds,
    recentIds,
    sampleCount,
  }: {
    hallOfFameIds: readonly string[];
    recentIds: readonly string[];
    sampleCount: number;
  }): readonly SnapshotSample[] {
    if (sampleCount <= 0) {
      return [];
    }

    const samples: SnapshotSample[] = [];
    const pool: readonly SnapshotSample[] = [
      ...hallOfFameIds.map((snapshotId): SnapshotSample => ({
        snapshotId,
        source: 'hall-of-fame',
      })),
      ...recentIds.map((snapshotId): SnapshotSample => ({
        snapshotId,
        source: 'recent',
      })),
    ];

    // Step 1: Guarantee at least one hall-of-fame sample when both pools exist.
    if (hallOfFameIds.length > 0 && recentIds.length > 0) {
      samples.push({
        snapshotId: hallOfFameIds[0] ?? '',
        source: 'hall-of-fame',
      });
    }

    // Step 2: Fill the remainder from the combined pool in round-robin order.
    let poolIndex = 0;
    while (samples.length < sampleCount && pool.length > 0) {
      const picked = pool[poolIndex % pool.length];
      if (picked !== undefined) {
        samples.push(picked);
      }
      poolIndex += 1;
    }

    return samples;
  }
}

// Re-export the core→race-pack snapshot adapter so callers that import from
// the opponent-snapshot service can convert pool snapshots to race-pack shape.
export { convertCoreToRacePackSnapshot } from './simulation-worker.race-pack.service';
