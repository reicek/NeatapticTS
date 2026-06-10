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
 * TODO: NGE_TODO — When NGE EpisodicSlot and GatingRouter primitives become
 * available (upstream Phase G/E), the snapshot payload format should be extended
 * to include episodic context and hard task-switch state.
 */

/** Snapshot update cadence configuration. */
export type OpponentSnapshotConfig = {
  /** Apply a snapshot update every this many generations. */
  readonly updateEveryNGenerations: number;
};

/** Rolling opponent snapshot store with generation barrier enforcement. */
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
}
