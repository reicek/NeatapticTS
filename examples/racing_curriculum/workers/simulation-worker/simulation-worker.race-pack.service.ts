import type { RacingRenderFrame } from './simulation-worker.types';

/**
 * Frozen opponent snapshot used as deterministic race-pack input.
 *
 * TODO: NGE_TODO — When NGE EpisodicSlot and GatingRouter primitives become
 * available (upstream Phase G/E), the snapshot payload format should be extended
 * to include episodic context and hard task-switch state.
 */
export type OpponentSnapshot = {
  /** Stable identifier frozen at snapshot capture time. */
  readonly snapshotId: string;
  /** Generation index at which this snapshot was captured. */
  readonly generation: number;
  /** Serialised network payloads for opponent controllers. */
  readonly networkPayloads: readonly unknown[];
};

/** Number of agent slots in a standard Tier-0 race pack. */
const AGENT_COUNT = 4 as const;

/** Grid column spacing in logical world units. */
const GRID_COLUMN_SPACING = 12 as const;

/** Grid row spacing in logical world units. */
const GRID_ROW_SPACING = 20 as const;

/** Seed scale factor for X offset per agent. */
const SEED_X_SCALE = 0.01 as const;

/** Expected typed-array buffer count (no pitStatus) for the standard pack. */
const EXPECTED_TRANSFER_BUFFER_COUNT = 10 as const;

/**
 * Constructs an initial race frame deterministically from a seed and a frozen
 * opponent snapshot.  Identical seed + identical snapshot → identical frame.
 *
 * Car positions are spread on a 2×2 grid offset by the seed so that no car
 * starts at the origin and positions are repeatable without an external PRNG.
 *
 * @param seed - Deterministic race-pack seed.
 * @param opponentSnapshot - Frozen opponent snapshot for the episode.
 * @returns Packed `RacingRenderFrame` with `schemaVersion: 'racing-packed-v1'`.
 *
 * @example
 * ```ts
 * const packA = createDeterministicRacePack(42, snapshot);
 * const packB = createDeterministicRacePack(42, snapshot);
 * // Array.from(packA.carX) deepEquals Array.from(packB.carX)
 * ```
 */
export function createDeterministicRacePack(
  seed: number,
  opponentSnapshot: OpponentSnapshot,
): RacingRenderFrame {
  // Step 1: Build deterministic per-car X and Y positions from seed + index.
  const carXPositions = buildDeterministicCarXPositions(seed);
  const carYPositions = buildDeterministicCarYPositions(seed, opponentSnapshot);

  // Step 2: Build the remaining per-car arrays.
  const carHeadings = new Float32Array(AGENT_COUNT);
  const carActive = new Uint8Array([1, 1, 1, 1]);
  const carTeam = new Uint8Array([0, 0, 1, 1]);
  const carMode = new Uint8Array(AGENT_COUNT);
  const tireState = new Float32Array(AGENT_COUNT * 4);
  const radioField = new Float32Array(0);
  const lap = new Uint16Array(AGENT_COUNT);
  const place = new Uint8Array([1, 2, 3, 4]);

  // Step 3: Assemble the packed frame.
  return {
    schemaVersion: 'racing-packed-v1',
    tick: 0,
    seed,
    trackId: 0,
    agentCount: AGENT_COUNT,
    featureFlags: 0,
    carX: carXPositions,
    carY: carYPositions,
    carHeading: carHeadings,
    carActive,
    carTeam,
    carMode,
    tireState,
    radioField,
    lap,
    place,
    raceTimeMs: 0,
    done: false,
  };

  /**
   * Builds deterministic X positions spread across a 2-column grid.
   * Position depends only on agent index and seed so identical inputs yield
   * identical outputs.
   *
   * @param raceSeed - Race-pack seed value.
   * @returns Float32Array of length AGENT_COUNT.
   */
  function buildDeterministicCarXPositions(raceSeed: number): Float32Array {
    const positions = new Float32Array(AGENT_COUNT);

    for (let agentIndex = 0; agentIndex < AGENT_COUNT; agentIndex++) {
      const columnIndex = agentIndex % 2;
      positions[agentIndex] =
        columnIndex * GRID_COLUMN_SPACING + raceSeed * SEED_X_SCALE;
    }

    return positions;
  }

  /**
   * Builds deterministic Y positions spread across grid rows.
   * Incorporates the snapshot generation to ensure pack distinctness between
   * different opponent snapshots.
   *
   * @param raceSeed - Race-pack seed value.
   * @param snapshot - Frozen opponent snapshot for the episode.
   * @returns Float32Array of length AGENT_COUNT.
   */
  function buildDeterministicCarYPositions(
    raceSeed: number,
    snapshot: OpponentSnapshot,
  ): Float32Array {
    const positions = new Float32Array(AGENT_COUNT);

    for (let agentIndex = 0; agentIndex < AGENT_COUNT; agentIndex++) {
      const rowIndex = Math.floor(agentIndex / 2);
      positions[agentIndex] =
        rowIndex * GRID_ROW_SPACING + (raceSeed + snapshot.generation) * SEED_X_SCALE;
    }

    return positions;
  }
}

/**
 * Collects every `ArrayBuffer` backing a typed-array field in the frame into a
 * transfer list for zero-copy `postMessage` transfer.
 *
 * Mirrors `resolveRacingRenderFrameTransferList` from the snapshot utils but is
 * owned by this service boundary so race-step streaming follows the same
 * zero-copy ownership contract.
 *
 * Rules:
 * - Every typed-array field contributes exactly one buffer entry.
 * - Shared buffers are deduplicated (listed only once).
 * - A standard pack without `pitStatus` produces exactly
 *   {@link EXPECTED_TRANSFER_BUFFER_COUNT} entries.
 *
 * @param frame - Packed render frame whose buffers will be transferred.
 * @returns Ordered list of `ArrayBuffer` references for postMessage transfer.
 *
 * @example
 * ```ts
 * const transferList = resolveRaceStepTransferList(pack);
 * worker.postMessage({ type: 'race-step', pack }, transferList);
 * ```
 */
export function resolveRaceStepTransferList(
  frame: RacingRenderFrame,
): ArrayBuffer[] {
  const transferList: ArrayBuffer[] = [];
  const seenBuffers = new Set<ArrayBuffer>();

  const typedArrayFields = [
    frame.carX,
    frame.carY,
    frame.carHeading,
    frame.carActive,
    frame.carTeam,
    frame.carMode,
    frame.tireState,
    frame.radioField,
    frame.lap,
    frame.place,
    ...(frame.pitStatus !== undefined ? [frame.pitStatus] : []),
  ];

  // Step 1: Walk fields; deduplicate shared buffers; collect in order.
  for (const typedArray of typedArrayFields) {
    const buffer = typedArray.buffer as ArrayBuffer;

    if (seenBuffers.has(buffer)) {
      continue;
    }

    seenBuffers.add(buffer);
    transferList.push(buffer);
  }

  return transferList;
}

export { EXPECTED_TRANSFER_BUFFER_COUNT };
