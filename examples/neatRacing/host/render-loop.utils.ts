/**
 * Host render-loop utilities for NeatRacing (slice P1-S1-D, plan
 * plans/neatracing-voxel-racer.plans.md, step 02, spec sections 5.1/6.3).
 *
 * This module owns the deterministic part of the NeatRacing render loop:
 * frame-delta clamping, the fixed-timestep accumulator that releases
 * simulation steps at a strict 60 Hz cadence, the spiral-of-death guard, and
 * the preallocated matrix/vector buffers that later render passes fill in.
 * The browser-entry glue (rAF scheduling, WebGL2 calls, input sampling)
 * lives in sibling slices and consumes these executors.
 *
 * Allocation policy (AC-042, render-loop share): every Float64Array is
 * allocated exactly once at state creation through the injectable allocator
 * seam. The per-frame executors only read/write plain numbers and the
 * preallocated buffers, so a steady-state render loop performs zero host
 * allocations per frame.
 *
 * Floating-point note (AC-003): `Math.floor(accumulatedMs / timestep)`
 * loses exactly one tick per simulated second when frame timestamps carry
 * ulp-scale rounding drift (for example `i * (1000 / 144)` timestamps), so
 * the step boundary is relaxed by an ulp-scale epsilon. The epsilon only
 * flips boundaries that differ by ~1e-9 ms (a nanosecond), which preserves
 * the spec cadence exactly while keeping the accumulator ulp-robust.
 */

/** Fixed simulation timestep in milliseconds: 60 simulation ticks per second (spec section 5.1). */
export const NEATRACING_FIXED_TIMESTEP_MS = 1000 / 60;

/**
 * Fixed simulation timestep in seconds, the unit physics and simulation
 * code consume (spec section 5.1).
 */
export const NEATRACING_FIXED_TIMESTEP_SECONDS = 1 / 60;

/**
 * Maximum number of simulation steps released in a single rAF frame. This is
 * the spiral-of-death guard: after a stall (tab switch, GC pause) the loop
 * plays at most 4 ticks per frame instead of freezing at 100% catch-up load
 * (spec section 6.3).
 */
export const NEATRACING_MAX_CATCHUP_STEPS_PER_FRAME = 4;

/**
 * Upper bound for a single clamped frame delta in milliseconds. Deltas above
 * this bound (tab switches, debugger pauses) are clamped so a stall cannot
 * pump a large backlog into the accumulator at once.
 */
export const NEATRACING_MAX_FRAME_DELTA_MS = 100;

/**
 * Accumulator ceiling in milliseconds: the maximum catch-up backlog the
 * accumulator ever holds (`max catchup steps * fixed timestep`). Excess
 * simulated time is dropped instead of accumulating into a spiral.
 */
export const NEATRACING_CATCHUP_CEILING_MS =
  NEATRACING_MAX_CATCHUP_STEPS_PER_FRAME * NEATRACING_FIXED_TIMESTEP_MS;

/**
 * Ulp-scale boundary epsilon in step units. It widens the floor boundary of
 * the step-count division so ulp-level floating-point drift in accumulated
 * deltas cannot swallow a step release that ideal frame streams are owed
 * (AC-003 exactness); it is far below any physical frame delta.
 */
export const NEATRACING_STEP_BOUNDARY_EPSILON = 1e-9;

/** Float length of a column-major 4x4 matrix. */
export const NEATRACING_RENDER_LOOP_MATRIX_FLOATS = 16;

/** Float length of a 3-component vector. */
export const NEATRACING_RENDER_LOOP_VECTOR_FLOATS = 3;

/**
 * Allocation seam for the render-loop's preallocated buffers. Browser-entry
 * glue injects the default allocator; tests inject a counting double to
 * assert the zero-per-frame-allocation contract (AC-042).
 */
export interface NeatRacingRenderLoopAllocator {
  /** Allocates a fresh Float64Array of the requested float length. */
  float64Array(length: number): Float64Array;
}

/**
 * Mutable accumulator state shared by the render-loop executors. The matrix
 * and vector buffers are preallocated once and filled in-place by the later
 * projection/view/camera passes each frame.
 */
export interface NeatRacingRenderLoopState {
  /** Simulated milliseconds not yet consumed by released steps. */
  accumulatedMs: number;
  /** Previous rAF timestamp, or null before the first frame. */
  lastTimestampMs: number | null;
  /** Preallocated column-major 4x4 projection matrix (16 floats). */
  projectionMatrix: Float64Array;
  /** Preallocated column-major 4x4 view matrix (16 floats). */
  viewMatrix: Float64Array;
  /** Preallocated camera position vector (3 floats). */
  cameraPosition: Float64Array;
  /** Preallocated camera forward-axis vector (3 floats). */
  cameraForward: Float64Array;
}

/**
 * Creates the default allocator that produces standard Float64Array buffers.
 *
 * @returns An allocator backed by plain Float64Array construction.
 *
 * @example
 * const allocator = createDefaultNeatRacingRenderLoopAllocator();
 * const state = createNeatRacingRenderLoopState(allocator);
 */
export function createDefaultNeatRacingRenderLoopAllocator(): NeatRacingRenderLoopAllocator {
  return {
    float64Array(length: number): Float64Array {
      return new Float64Array(length);
    },
  };
}

/**
 * Creates the render-loop state, allocating every matrix and vector exactly
 * once through the given allocator seam.
 *
 * @param allocator - Allocation seam; defaults to the standard allocator.
 * @returns The initial accumulator state with preallocated buffers.
 *
 * @example
 * const state = createNeatRacingRenderLoopState();
 * const steps = advanceFrameSteps(state, timestampMs);
 */
export function createNeatRacingRenderLoopState(
  allocator: NeatRacingRenderLoopAllocator = createDefaultNeatRacingRenderLoopAllocator(),
): NeatRacingRenderLoopState {
  return {
    accumulatedMs: 0,
    lastTimestampMs: null,
    projectionMatrix: allocator.float64Array(
      NEATRACING_RENDER_LOOP_MATRIX_FLOATS,
    ),
    viewMatrix: allocator.float64Array(NEATRACING_RENDER_LOOP_MATRIX_FLOATS),
    cameraPosition: allocator.float64Array(
      NEATRACING_RENDER_LOOP_VECTOR_FLOATS,
    ),
    cameraForward: allocator.float64Array(NEATRACING_RENDER_LOOP_VECTOR_FLOATS),
  };
}

/**
 * Computes the clamped frame delta between the previous and current rAF
 * timestamps (recycled verbatim from the Neatenstein render-loop precedent).
 * The first frame (null previous timestamp) contributes a 0 ms delta so the
 * accumulator cannot pre-fill before real frames arrive.
 *
 * @param lastTimestampMs - Previous rAF timestamp, or null on the first frame.
 * @param timestampMs - Current rAF timestamp.
 * @param maxDeltaMs - Clamp bound for oversized deltas (tab switches, pauses).
 * @returns The clamped non-negative delta in milliseconds.
 *
 * @example
 * computeDeltaMs(null, 16.6, 100); // 0   (first frame)
 * computeDeltaMs(0, 5000, 100);    // 100 (clamped tab switch)
 * computeDeltaMs(0, 16.6, 100);    // 16.6 (normal frame)
 */
export function computeDeltaMs(
  lastTimestampMs: number | null,
  timestampMs: number,
  maxDeltaMs: number,
): number {
  return Math.min(
    lastTimestampMs === null ? 0 : timestampMs - lastTimestampMs,
    maxDeltaMs,
  );
}

/**
 * Advances the accumulator for one rAF frame and returns how many simulation
 * steps the loop must run this frame. Steps are released from the fixed
 * 60 Hz cadence accumulator, capped at `NEATRACING_MAX_CATCHUP_STEPS_PER_FRAME`
 * (spiral-of-death guard), and excess backlog above the catch-up ceiling is
 * dropped so the loop cannot fall behind permanently. Sub-timestep frames
 * release 0 steps (never a forced minimum of 1) and carry their remainder to
 * the next frame. The call mutates `state` in place and performs no heap
 * allocations (AC-042).
 *
 * @param state - Render-loop state mutated in place.
 * @param timestampMs - Current rAF timestamp.
 * @returns The number of simulation steps to run this frame (0 to 4).
 *
 * @example
 * const state = createNeatRacingRenderLoopState();
 * const stepCount = advanceFrameSteps(state, performance.now());
 * for (let stepIndex = 0; stepIndex < stepCount; stepIndex++) {
 *   // run one 1/60 s simulation tick
 * }
 */
export function advanceFrameSteps(
  state: NeatRacingRenderLoopState,
  timestampMs: number,
): number {
  const deltaMs = computeDeltaMs(
    state.lastTimestampMs,
    timestampMs,
    NEATRACING_MAX_FRAME_DELTA_MS,
  );
  state.lastTimestampMs = timestampMs;
  state.accumulatedMs = Math.min(
    state.accumulatedMs + deltaMs,
    NEATRACING_CATCHUP_CEILING_MS,
  );
  const stepCount = Math.min(
    Math.floor(
      state.accumulatedMs / NEATRACING_FIXED_TIMESTEP_MS +
        NEATRACING_STEP_BOUNDARY_EPSILON,
    ),
    NEATRACING_MAX_CATCHUP_STEPS_PER_FRAME,
  );
  state.accumulatedMs -= stepCount * NEATRACING_FIXED_TIMESTEP_MS;
  return stepCount;
}
