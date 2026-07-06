/**
 * Racing evaluation-pack normalizer (Layer 3 wrapper around Layer 2 generic pack).
 *
 * Provides the benchmark-local `populateRacingFrame` seam that consumes a core
 * `DeterministicEvaluationPack` and produces a `RacingRenderFrame`. Also exposes
 * thin racing-specific helpers for transfer-list resolution and schema-version
 * assertion that delegate to the generic Layer 2 contracts.
 *
 * The generic pack is intentionally transport-neutral: it only promises
 * determinism for the tuple `(seed, agentCount, schemaVersion)`. The racing
 * wrapper adds the benchmark-local reproducibility tuple
 * `(seed, agentCount, packSchemaVersion, opponentSnapshot, trackId, featureFlags)`.
 * Same full tuple → identical `RacingRenderFrame` on the same runtime. This
 * preserves Layer 2 reuse while letting racing own its own frame format and
 * lifecycle.
 *
 * Array mapping invariant: the generic pack exposes `[agentStates,
 * agentWeights, agentActive]` in that order. The racing wrapper maps these
 * deterministically to `carX`, `carY`, `carHeading`, `carActive`, and
 * `carTeam` using pure functions, so the same source arrays always produce the
 * same frame fields.
 *
 * @module simulation-worker.evaluation-pack.normalizer
 */

import type { DeterministicEvaluationPack } from '../../../../src/architecture/network/evaluation-pack/network.evaluation-pack';
import type { RacingRenderFrame } from './simulation-worker.types';
import { resolveRacingRenderFrameTransferList } from './simulation-worker.snapshot.utils';

/** Racing-specific inputs that accompany a generic deterministic pack. */
export type RacingEvaluationConfig = {
  /** Frozen opponent snapshot for the episode. */
  readonly opponentSnapshot: {
    readonly snapshotId: string;
    readonly generation: number;
    readonly networkPayloads: readonly unknown[];
  };
  /** Identifies the frozen TrackSpec this frame belongs to. */
  readonly trackId: number;
  /** Feature-flag bitfield: bit 0 = tiresEnabled, bit 1 = radioEnabled, bit 2 = pitsEnabled. */
  readonly featureFlags: number;
  /** Number of agent slots (fixed for the episode). */
  readonly agentCount: number;
  /** Expected generic schema version for the incoming pack. */
  readonly packSchemaVersion: string;
};

/**
 * Populates a `RacingRenderFrame` from a generic deterministic evaluation pack
 * and racing-specific configuration.
 *
 * Racing reproducibility tuple:
 * `(seed, agentCount, packSchemaVersion, opponentSnapshot, trackId, featureFlags)`.
 * Same tuple → identical racing frame on the same runtime.
 *
 * The deterministic array mapping is: `agentStates → carX/carHeading`,
 * `agentWeights → carY`, `agentActive → carActive/carTeam`. Because the mapping
 * uses pure functions (copy constructors and modulo), the same generic pack and
 * racing configuration always produce the same frame fields.
 *
 * @param pack - Generic deterministic evaluation pack from Layer 2.
 * @param racingConfig - Racing-specific frame inputs.
 * @returns A packed `RacingRenderFrame` with `schemaVersion: 'racing-packed-v1'`.
 * @throws {RangeError} When the pack schema version or agent count does not
 *   match `racingConfig`.
 *
 * @example
 * ```ts
 * const pack = createDeterministicEvaluationPack(42, {
 *   agentCount: 4,
 *   schemaVersion: 'eval-pack-v1',
 * });
 * const frame = populateRacingFrame(pack, RACING_CONFIG);
 * console.log(frame.schemaVersion); // 'racing-packed-v1'
 * ```
 */
export function populateRacingFrame(
  pack: DeterministicEvaluationPack,
  racingConfig: RacingEvaluationConfig,
): RacingRenderFrame {
  // Step 1: Validate the generic pack schema version against racing expectation
  assertRacingPackSchemaVersion(pack, racingConfig.packSchemaVersion);

  // Step 2: Ensure the pack's typed-array dimensions match the racing agent count
  validatePackAgentCount(pack, racingConfig.agentCount);

  // Step 3: Map the core pack arrays to racing frame fields deterministically
  const [agentStates, agentWeights, agentActive] = pack.arrays;
  const agentCount = racingConfig.agentCount;

  const carX = Float32Array.from(agentStates as Float32Array);
  const carY = Float32Array.from(
    agentWeights as Float64Array,
    (weight) => weight,
  );
  const carHeading = Float32Array.from(agentStates as Float32Array);
  const carActive = Uint8Array.from(agentActive as Uint8Array);
  const carTeam = Uint8Array.from(
    { length: agentCount },
    (_, index) => (agentActive as Uint8Array)[index] % 2,
  );
  const carMode = Uint8Array.from({ length: agentCount }, (_, index) =>
    (agentActive as Uint8Array)[index] > 127 ? 1 : 0,
  );
  const tireState = Float32Array.from(
    { length: agentCount * 4 },
    (_, index) => (agentStates as Float32Array)[Math.floor(index / 4)],
  );
  const radioField = new Float32Array(0);
  const lap = Uint16Array.from(
    { length: agentCount },
    (_, index) => (agentActive as Uint8Array)[index],
  );
  const place = Uint8Array.from(
    { length: agentCount },
    (_, index) => (agentActive as Uint8Array)[index],
  );

  // Step 4: Assemble the racing render frame
  return {
    schemaVersion: 'racing-packed-v1',
    tick: 0,
    seed: pack.seed,
    trackId: racingConfig.trackId,
    agentCount,
    featureFlags: racingConfig.featureFlags,
    carX,
    carY,
    carHeading,
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
}

/**
 * Resolves the zero-copy postMessage transfer list for a racing render frame.
 *
 * Delegates to the racing-specific transfer-list resolver so the Layer 3
 * normalizer and the existing snapshot utilities share the same zero-copy
 * ownership contract. The transfer list follows the HTML structured-clone
 * transferables contract consumed by Web Workers; for background see MDN Web
 * Docs,
 * [Transferable objects](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Transferring_objects_from_one_worker_to_another).
 *
 * @param frame - Packed racing render frame.
 * @returns Ordered list of distinct `ArrayBuffer` references.
 *
 * @example
 * ```ts
 * const transferList = resolveRacingTransferList(frame);
 * worker.postMessage({ type: 'render', frame }, transferList);
 * ```
 */
export function resolveRacingTransferList(
  frame: RacingRenderFrame,
): ArrayBuffer[] {
  return resolveRacingRenderFrameTransferList(frame);
}

/**
 * Asserts that the incoming generic pack's schema version matches the racing
 * configuration's expected version.
 *
 * This is the racing-specific wrapper around the generic
 * `assertSchemaVersion` boundary guard. Rejecting an unexpected version at
 * the boundary prevents silent misinterpretation of packed bytes. For background
 * on schema-version sentinels and forward compatibility, see Wikipedia
 * contributors,
 * [Forward compatibility](https://en.wikipedia.org/wiki/Forward_compatibility).
 *
 * @param pack - Object with a `schemaVersion` field.
 * @param expectedVersion - Expected schema-version sentinel.
 * @throws {RangeError} When the schema version does not match.
 *
 * @example
 * ```ts
 * assertRacingPackSchemaVersion({ schemaVersion: 'eval-pack-v1' }, 'eval-pack-v1');
 * // no error
 * ```
 */
export function assertRacingPackSchemaVersion(
  pack: { schemaVersion: unknown },
  expectedVersion: string,
): void {
  if (pack.schemaVersion !== expectedVersion) {
    throw new RangeError(
      `Unsupported evaluation-pack schemaVersion: expected '${expectedVersion}', got '${String(pack.schemaVersion)}'`,
    );
  }
}

// ---------------------------------------------------------------------------
// Helpers (below the fold)
// ---------------------------------------------------------------------------

// Validates that every typed array in the generic pack has the expected
// agent-count dimension. Racing frame fields are sized from
// racingConfig.agentCount, so a dimension mismatch would silently corrupt
// per-car state if it were allowed through.
function validatePackAgentCount(
  pack: DeterministicEvaluationPack,
  expectedAgentCount: number,
): void {
  const dimensions = pack.arrays.map(
    (array) => (array as ArrayBufferView & { length: number }).length,
  );
  const hasMismatch = dimensions.some(
    (length) => length !== expectedAgentCount,
  );

  if (dimensions.length === 0 || hasMismatch) {
    throw new RangeError(
      `Agent count mismatch: racingConfig.agentCount=${expectedAgentCount}, pack array dimensions=[${dimensions.join(', ')}]`,
    );
  }
}
