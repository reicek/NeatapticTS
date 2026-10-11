/**
 * Save→rebuild pipeline for the NeatRacing editor (slice P1-S6-E; spec
 * section 8.3 policy P5, AC-041): every {@linkcode EditorRebuildPipeline.edit}
 * emits an immediate dirty-slice remesh covering the edited z-slice's clamped,
 * deduplicated AABB ± 1 window, then arms a
 * {@linkcode NEATRACING_REBUILD_DEBOUNCE_MS} trailing debounce that coalesces
 * the drag burst into exactly one full rebuild, and
 * {@linkcode EditorRebuildPipeline.pointerUp} flushes that pending debounce
 * into one immediate full rebuild.
 *
 * The policy is latest-wins single-flight (spec P5 bullet 2): only one rebuild
 * is ever in flight, work requested mid-flight is queued behind the in-flight
 * rebuild (its superseded result is discarded; the
 * {@linkcode EditorRebuildPipeline.snapshot} `supersededCount` counter is
 * arrival-time), the queue drains strictly in FIFO order, and full-rebuild
 * durations measured through the injected `nowMs` clock yield the nearest-rank
 * `p95Ms` statistic the contract tests hold against the
 * {@linkcode NEATRACING_REBUILD_P95_BUDGET_MS} budget.
 *
 * The module is pure scheduling/state — no DOM, no WebGL, and no RNG — so the
 * pipeline is exactly deterministic: the same edit/pointer-up sequence and
 * injected clock in, the same rebuild call order and snapshot out on every
 * run and runtime.
 */

import { NEATRACING_EDITOR_VOLUME_EDGE } from './editor.state';

/** Highest inclusive slice z on the editor volume: 128 - 1 = 127. */
const NEATRACING_EDITOR_MAX_SLICE = NEATRACING_EDITOR_VOLUME_EDGE - 1;

/** Trailing debounce window before a drag burst coalesces into one full rebuild (ms). */
export const NEATRACING_REBUILD_DEBOUNCE_MS = 250;

/** p95 budget ceiling for measured full-rebuild durations (ms). */
export const NEATRACING_REBUILD_P95_BUDGET_MS = 300;

/** Nearest-rank quantile location for the p95 of full-rebuild durations. */
const P95_RANK_QUANTILE = 0.95;

/** One rebuild workload emitted or queued by the pipeline. */
export interface EditorRebuildRequest {
  /** Rebuild kind: `dirty` remeshes a slice window, `full` rebuilds everything. */
  mode: 'dirty' | 'full';
  /** Dirty-window slice indices in ascending order; full requests carry none. */
  slices: readonly number[];
}

/** Rebuild executor injected into the pipeline by the editor integration. */
type EditorRebuildExecutor = (request: EditorRebuildRequest) => Promise<void>;

/** Monotonic clock accessor used to measure full-rebuild durations. */
type EditorRebuildClock = () => number;

/** Factory configuration for the editor rebuild pipeline. */
export interface EditorRebuildPipelineOptions {
  /** Rebuild executor invoked for every dirty remesh and full rebuild. */
  rebuild: EditorRebuildExecutor;
  /**
   * Monotonic clock accessor for full-rebuild duration measurement; defaults
   * to the ambient high-resolution relative clock when omitted.
   */
  nowMs?: EditorRebuildClock;
}

/** Read-only scheduling and timing statistics for one pipeline moment. */
interface EditorRebuildSnapshot {
  /** True while a full rebuild is occupying the single rebuild slot. */
  inFlight: boolean;
  /** True while at least one queued request waits behind the in-flight rebuild. */
  queued: boolean;
  /** Saves that arrived while a rebuild was in flight (latest-wins count). */
  supersededCount: number;
  /** Measured full-rebuild durations (only completions, in completion order). */
  durationsMs: number[];
  /** Nearest-rank 95th percentile of {@linkcode EditorRebuildSnapshot.durationsMs}. */
  p95Ms: number;
}

/** Scheduler surface returned by {@linkcode createEditorRebuildPipeline}. */
export interface EditorRebuildPipeline {
  /**
   * Emits one immediate dirty-slice remesh for the edited z-slice's clamped,
   * deduplicated AABB ± 1 window, counts a superseded result when a rebuild
   * is already in flight (the request queues behind it), and rearms the
   * trailing debounce so a drag burst coalesces into one full rebuild.
   *
   * @param sliceZ - Edited slice coordinate, clamped to the editor volume.
   * @returns Nothing; rebuild work is dispatched through the injected
   * executor.
   */
  edit(sliceZ: number): void;
  /**
   * Consumes the pending debounce and flushes one immediate full rebuild.
   * Any accepted-not-yet-run work still drains first in FIFO order.
   *
   * @returns Nothing; the flush dispatches through the injected executor.
   */
  pointerUp(): void;
  /**
   * Read-only snapshot of the single-flight scheduling state and the
   * nearest-rank p95 of the measured full-rebuild durations.
   *
   * @returns A frozen copy of the scheduling statistics at call time.
   */
  snapshot(): EditorRebuildSnapshot;
}

/**
 * Ambient clock fallback for {@linkcode EditorRebuildPipelineOptions.nowMs}:
 * Node's high-resolution relative milliseconds stay monotonic across the
 * duration-measured requests, so recorded durations are order-independent.
 *
 * @returns Elapsed high-resolution milliseconds relative to process start.
 */
function defaultNowMs(): number {
  return performance.now();
}

/**
 * Clamps one slice coordinate into the editor volume window 0..127
 * ({@linkcode NEATRACING_EDITOR_VOLUME_EDGE} - 1 inclusive on both ends).
 *
 * @param sliceZ - Unclamped slice coordinate input.
 * @returns The slice coordinate pinned into the editor volume window.
 */
function clampSliceZ(sliceZ: number): number {
  return Math.min(Math.max(sliceZ, 0), NEATRACING_EDITOR_MAX_SLICE);
}

/**
 * Builds the dirty-slice remesh request for one edited slice: the clamped
 * AABB ± 1 window `[z - 1, z, z + 1]`, deduplicated where the clamped edges
 * pin against the volume boundary (0 → `[0, 1]`, 127 → `[126, 127]`).
 *
 * @param sliceZ - Edited slice coordinate, clamped before windowing.
 * @returns The dirty request carrying the ascending deduplicated window.
 */
function buildDirtyRequest(sliceZ: number): EditorRebuildRequest {
  const clipped = clampSliceZ(sliceZ);
  const window = Array.from(
    new Set([clampSliceZ(clipped - 1), clipped, clampSliceZ(clipped + 1)]),
  );
  return { mode: 'dirty', slices: window };
}

/**
 * Nearest-rank percentile of one duration list: the smallest value whose
 * cumulative rank reaches `quantile * length`. An empty list reports 0 so a
 * parked-first-rebuild snapshot still yields a well-formed 0 ms statistic
 * before any rebuild completes.
 *
 * @param values - Completed durations in any order.
 * @returns The nearest-rank quantile in the same millisecond units.
 */
function percentileNearestRank(values: readonly number[]): number {
  if (values.length === 0) {
    return 0;
  }
  const sorted = values.toSorted((a, b) => a - b);
  const rank = Math.ceil(P95_RANK_QUANTILE * sorted.length);
  return sorted[rank - 1];
}

/**
 * Pipeline factory: binds one rebuild executor and an optional monotonic
 * clock into a mutable edit→rebuild scheduler that serializes through a
 * single FIFO queue, so callers never observe two rebuilds in flight.
 *
 * @param options - Executor plus optional duration clock (see
 * {@linkcode EditorRebuildPipelineOptions}).
 * @returns The `{@linkcode EditorRebuildPipeline.edit}`,
 * `{@linkcode EditorRebuildPipeline.pointerUp}`,
 * `{@linkcode EditorRebuildPipeline.snapshot}` scheduler surface.
 * @example
 * ```ts
 * const pipeline = createEditorRebuildPipeline({
 *   rebuild: async () => remeshDirtySlices(),
 * });
 * pipeline.edit(42); // immediate [41, 42, 43] remesh + 250 ms debounce
 * pipeline.pointerUp(); // pending debounce consumed → immediate full
 * const stats = pipeline.snapshot(); // in-flight/queued/p95 statistics
 * ```
 */
export function createEditorRebuildPipeline(
  options: EditorRebuildPipelineOptions,
): EditorRebuildPipeline {
  const { rebuild, nowMs = defaultNowMs } = options;

  /** Pending debounce timer handle; rearmed by every edit. */
  let timerId: ReturnType<typeof setTimeout> | undefined;
  /** Full rebuilds currently occupying the single rebuild slot. */
  let runningFullCount = 0;
  /** Latest-wins counter for saves arriving while a rebuild is in flight. */
  let supersededCount = 0;
  /** Drain latch: the queue loop is the only concurrent runner. */
  let draining = false;
  /** Accepted requests waiting behind the in-flight rebuild (FIFO). */
  const queue: EditorRebuildRequest[] = [];
  /** Completed full-rebuild durations, in completion order. */
  const durationsMs: number[] = [];

  /**
   * Queues one request and kicks the drain loop; on an idle pipeline the
   * executor is invoked before this returns, on a busy one the request
   * waits behind the in-flight rebuild.
   *
   * @param request - Accepted remesh or rebuild workload.
   */
  function enqueue(request: EditorRebuildRequest): void {
    queue.push(request);
    void drain();
  }

  /**
   * Single-runner drain loop: shifts queued requests in FIFO order until the
   * queue empties. Re-entrant kicks (enqueue while draining) return without
   * starting a second loop, so at most one rebuild runs at a time.
   */
  async function drain(): Promise<void> {
    if (draining) {
      return;
    }
    draining = true;
    while (queue.length > 0) {
      const request = queue.shift();
      if (request === undefined) {
        break;
      }
      try {
        await run(request);
      } catch {
        // A rejecting executor must not wedge the queue; the failed
        // rebuild's result is discarded per the latest-wins policy.
      }
    }
    draining = false;
  }

  /**
   * Tracks full-mode flight state around one executor call and records the
   * measured duration for every completed full rebuild.
   *
   * @param request - Accepted request handed to the executor.
   */
  async function run(request: EditorRebuildRequest): Promise<void> {
    const isFull = request.mode === 'full';
    if (isFull) {
      runningFullCount += 1;
    }
    const startedAtMs = nowMs();
    try {
      await rebuild(request);
    } finally {
      if (isFull) {
        runningFullCount -= 1;
        durationsMs.push(nowMs() - startedAtMs);
      }
    }
  }

  /**
   * Rearms the trailing debounce: the timer restarts on every accepted edit,
   * so a drag burst coalesces into exactly one full rebuild 250 ms after the
   * last edit.
   */
  function scheduleDebounce(): void {
    clearTimeout(timerId);
    timerId = setTimeout(() => {
      timerId = undefined;
      enqueue({ mode: 'full', slices: [] });
    }, NEATRACING_REBUILD_DEBOUNCE_MS);
  }

  const pipeline: EditorRebuildPipeline = {
    edit(sliceZ) {
      enqueue(buildDirtyRequest(sliceZ));
      if (runningFullCount > 0) {
        supersededCount += 1;
      }
      scheduleDebounce();
    },
    pointerUp() {
      clearTimeout(timerId);
      timerId = undefined;
      enqueue({ mode: 'full', slices: [] });
    },
    snapshot() {
      return {
        inFlight: runningFullCount > 0,
        queued: queue.length > 0,
        supersededCount,
        durationsMs: [...durationsMs],
        p95Ms: percentileNearestRank(durationsMs),
      };
    },
  };

  return pipeline;
}
