/**
 * Sim ↔ eval worker message contracts for the neatRacing in-band live
 * adaptation split (Phase 2, Step 02, slice P2-S2-A — tdd_sequence
 * `green-only`).
 *
 * Spec anchors:
 * - §9.2 two-worker split: the simulation worker owns 60 Hz physics and
 *   per-tick control (E11(a)); this protocol carries every growth/variant
 *   evaluation request to the dedicated eval worker (E11(b)), whose
 *   replies the sim side queues and applies only at deterministic
 *   adaptation-tick boundaries — never on reply arrival (E11(c)).
 * - E11(d): eval requests are self-contained per evaluation window. The
 *   payload carries the AC-026 tuple seed and the sim tick the request was
 *   issued at, and every result echoes both. The echo is what lets the
 *   sim-side queue drop superseded-round replies after a round-boundary
 *   `stop`, so no reply from a cancelled round can mutate a new round.
 * - D2: variant RNG streams are derived by the eval worker itself from the
 *   payload tuple seed (§9.7.2 row-4 composition via the recycled
 *   reproducibility table in ../simulation-worker/reproducibility.utils).
 *   The eval worker NEVER consumes or advances sim-worker network RNG
 *   state, so the protocol ships serialized payloads only.
 * - TP-18: live-mode variant dispatch is bounded by the named constant
 *   NEATRACING_EVAL_WORKER_PARALLEL_VARIANT_COUNT below — replacing the
 *   racing-curriculum demo's unbounded parallelVariantCount: 256.
 *
 * Realm note: this module is deliberately realm-free (no `self`, no
 * `postMessage`): the sim and eval sides share these same contracts, and
 * Jest drives them headless. The worker-realm glue lives in
 * ./eval-worker.ts — the eval side of the wire (session, bootstrap unit,
 * message adapter) — so both workers stay independently testable.
 */

/**
 * Live-mode parallel variant evaluation bound (TP-18).
 *
 * Educational note: the racing-curriculum demo evaluated up to 256 weight
 * variants in a single Promise.all window
 * (examples/racing_curriculum/browser-entry/browser-entry.ts:
 * `parallelVariantCount: 256`). In-band live mode works inside the 2 ms
 * adaptation-tick budget (TP-18: NEATRACING_ADAPTATION_TICK_BUDGET_MS),
 * so the eval worker dispatches at most 16 concurrent variant evaluations
 * — mirroring the library default
 * DEFAULT_ACCELERATION_PARALLEL_VARIANT_COUNT in
 * src/acceleration/acceleration.constants.ts — and processes larger
 * request batches as sequential bounded windows via runBoundedConcurrency.
 */
export const NEATRACING_EVAL_WORKER_PARALLEL_VARIANT_COUNT = 16;

/**
 * One variant slot inside an eval request (D2: eval-owned, serialized).
 *
 * The growth engine consumes these slots in Step 03; for now the protocol
 * pins only their transport shape so the sim side can be wired against a
 * stable contract.
 */
export interface NeatRacingEvalWorkerVariant {
  /** Variant slot index (non-negative integer) for the §9.7.2 stride table. */
  variantIndex: number;
  /**
   * Patch size the slot evaluates. Drives the per-variant seed stride
   * (stride = max(NEATRACING_VARIANT_PATCH_SEED_OFFSET,
   * max(1, patchSize) × NEATRACING_VARIANT_PATCH_SEED_STRIDE_FACTOR));
   * omitted means the patch size is not yet known and the recycled offset
   * base applies (patchSize 0).
   */
  patchSize?: number;
  /**
   * Opaque serialized variant payload (structured-clone-safe carrier the
   * growth engine consumes). The protocol transports it verbatim and never
   * inspects it — mirroring the sim worker's stance on opponentSnapshot.
   */
  snapshot: unknown;
}

/** sim → eval request payload: one bounded evaluation window. */
export interface NeatRacingEvalWorkerEvalPayload {
  /** AC-026 tuple seed driving every variant stream derivation. */
  seed: number;
  /**
   * Sim tick the request was issued at. Echoed on every result so the
   * sim-side queue applies replies only at deterministic boundaries at or
   * after this tick (E11(c)) — never on wall-clock reply arrival.
   */
  requestTick: number;
  /** Variant slots to evaluate under the bounded parallel dispatch. */
  variants: readonly NeatRacingEvalWorkerVariant[];
}

/** One scored variant slot, reported in request order. */
export interface NeatRacingEvalWorkerVariantResult {
  /** Variant slot index (echo of the request entry). */
  variantIndex: number;
  /** Deterministic fitness score produced by the evaluation unit. */
  score: number;
}

/** eval → sim result payload for one evaluation window. */
export interface NeatRacingEvalWorkerResultPayload {
  /** Echo of the request's AC-026 tuple seed (round identity). */
  seed: number;
  /** Echo of the request tick (the deterministic-boundary hook). */
  requestTick: number;
  /** Scores in request order (stable across replays of the same request). */
  results: readonly NeatRacingEvalWorkerVariantResult[];
}

/** eval → sim error payload (contract-level rejections). */
export interface NeatRacingEvalWorkerErrorPayload {
  /** Rejection reason (static contract vocabulary for easy assertions). */
  message: string;
}

/** sim → eval request union. */
export type NeatRacingEvalWorkerRequest =
  { type: 'eval'; payload: NeatRacingEvalWorkerEvalPayload } | { type: 'stop' };

/** eval → sim response union. */
export type NeatRacingEvalWorkerResponse =
  | { type: 'result'; payload: NeatRacingEvalWorkerResultPayload }
  | { type: 'error'; payload: NeatRacingEvalWorkerErrorPayload };

/** Validation failure carrier mirroring the sim worker's `{ error }` shape. */
export type NeatRacingEvalWorkerValidationError = { error: string };

/**
 * Validates one eval request payload (contract-level, static messages).
 *
 * Returns a rejection-reason record when the payload violates the eval
 * protocol, or null when it is well-formed. Mirrors the sim worker's
 * validation-helper style so both workers reject malformed traffic with
 * their single `error` reply.
 *
 * Validation surface: seed must be a finite number; requestTick must be a
 * non-negative integer; variants must be a non-empty array of objects each
 * carrying a non-negative integer variantIndex, an optional non-negative
 * integer patchSize, and a present (opaque) snapshot. The snapshot CONTENT
 * is deliberately unvalidated — it is the growth engine's input (Step 03),
 * and the protocol only guarantees transport (D2).
 *
 * @param payload - The decoded `eval` request payload as received on the
 *   wire (untyped on purpose: the contract guards structured-clone traffic).
 * @returns A rejection record, or null when the payload is well-formed.
 */
export function validateNeatRacingEvalWorkerEvalPayload(
  payload: unknown,
): NeatRacingEvalWorkerValidationError | null {
  // Step 1: Reject non-object payloads wholesale (arrays included).
  if (
    payload === null ||
    typeof payload !== 'object' ||
    Array.isArray(payload)
  ) {
    return { error: 'eval worker eval failed: payload must be an object' };
  }
  const candidate = payload as Partial<NeatRacingEvalWorkerEvalPayload>;

  // Step 2: Validate the deterministic identity fields.
  if (typeof candidate.seed !== 'number' || !Number.isFinite(candidate.seed)) {
    return { error: 'eval worker eval failed: seed must be a finite number' };
  }
  if (
    typeof candidate.requestTick !== 'number' ||
    !Number.isInteger(candidate.requestTick) ||
    candidate.requestTick < 0
  ) {
    return {
      error:
        'eval worker eval failed: requestTick must be a non-negative integer',
    };
  }

  // Step 3: Validate the variant batch (non-empty; every slot well-formed).
  const variants = candidate.variants;
  if (!Array.isArray(variants) || variants.length === 0) {
    return {
      error: 'eval worker eval failed: variants must be a non-empty array',
    };
  }
  for (let index = 0; index < variants.length; index += 1) {
    const variant = variants[index] as unknown;
    if (
      variant === null ||
      typeof variant !== 'object' ||
      Array.isArray(variant)
    ) {
      return {
        error: `eval worker eval failed: variants[${index}] must be an object`,
      };
    }
    const slot = variant as Partial<NeatRacingEvalWorkerVariant>;
    if (
      typeof slot.variantIndex !== 'number' ||
      !Number.isInteger(slot.variantIndex) ||
      slot.variantIndex < 0
    ) {
      return {
        error: `eval worker eval failed: variants[${index}].variantIndex must be a non-negative integer`,
      };
    }
    if (
      slot.patchSize !== undefined &&
      (typeof slot.patchSize !== 'number' ||
        !Number.isInteger(slot.patchSize) ||
        slot.patchSize < 0)
    ) {
      return {
        error: `eval worker eval failed: variants[${index}].patchSize must be a non-negative integer`,
      };
    }
    if (!('snapshot' in variant)) {
      return {
        error: `eval worker eval failed: variants[${index}].snapshot is required`,
      };
    }
  }
  return null;
}

/**
 * A lazy evaluation task factory: invoked only when a dispatch slot frees.
 */
export type NeatRacingEvalWorkerTask<T> = () => T | Promise<T>;

/**
 * Runs lazy task factories under a bounded parallel dispatch and resolves
 * the results in input order.
 *
 * Educational note: mirrors the neatenstein bounded-concurrency harness
 * (examples/neatenstein/browser-entry/harness/bounded-concurrency.ts) so
 * both examples keep the same vocabulary without a cross-example import.
 * Concurrency never exceeds `parallelCount` (TP-18: live-mode dispatch
 * stays inside the 2 ms adaptation-tick budget) and larger batches run as
 * sequential bounded windows. The chain never reads the clock and never
 * mutates timers, so fake-timer tests install freely.
 *
 * @param tasks - Lazy factories; each is invoked at most once, only when a
 *   slot is free.
 * @param parallelCount - Dispatch bound; defaults to
 *   NEATRACING_EVAL_WORKER_PARALLEL_VARIANT_COUNT.
 * @returns The results indexed exactly like the input tasks.
 * @throws {RangeError} When `parallelCount` is not a positive integer.
 */
export async function runBoundedConcurrency<T>(
  tasks: readonly NeatRacingEvalWorkerTask<T>[],
  parallelCount: number = NEATRACING_EVAL_WORKER_PARALLEL_VARIANT_COUNT,
): Promise<T[]> {
  if (!Number.isInteger(parallelCount) || parallelCount < 1) {
    throw new RangeError(
      `neatracing eval protocol failed: parallelCount must be a positive integer, received ${String(parallelCount)}`,
    );
  }
  const results = new Array<T>(tasks.length);
  let nextIndex = 0;
  const runSlot = async (): Promise<void> => {
    while (nextIndex < tasks.length) {
      const index = nextIndex;
      nextIndex += 1;
      results[index] = await tasks[index]();
    }
  };
  await Promise.all(
    Array.from({ length: Math.min(parallelCount, tasks.length) }, () =>
      runSlot(),
    ),
  );
  return results;
}
