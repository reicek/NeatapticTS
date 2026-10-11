/**
 * RED->GREEN contract tests for the NeatRacing host render-loop utilities
 * (slice P1-S1-D, plan plans/neatracing-voxel-racer.plans.md, step 02).
 *
 * Covered spec contracts:
 * - Spec sections 5.1/6.3 cadence constants: 60 Hz fixed timestep and a
 *   catch-up ceiling of at most 4 simulated steps per rAF frame.
 * - AC-003: the accumulator releases exactly 60 simulation ticks per
 *   simulated second for ideal frame streams at 60/120/144 Hz display
 *   cadences, with 0 or 1+ steps per frame (never a forced minimum of 1).
 * - AC-042 (render-loop share): preallocated matrices and vectors are
 *   allocated once through the allocator seam and reused for every frame,
 *   verified with a counting allocator double.
 *
 * The executors under test are pure: they never touch the DOM, rAF, or
 * WebGL2, so they run in the plain Node jest environment.
 */

import {
  NEATRACING_FIXED_TIMESTEP_MS,
  NEATRACING_FIXED_TIMESTEP_SECONDS,
  NEATRACING_MAX_CATCHUP_STEPS_PER_FRAME,
  NEATRACING_RENDER_LOOP_MATRIX_FLOATS,
  NEATRACING_RENDER_LOOP_VECTOR_FLOATS,
  advanceFrameSteps,
  computeDeltaMs,
  createNeatRacingRenderLoopState,
  type NeatRacingRenderLoopAllocator,
} from './render-loop.utils';

/** Counting allocator double that records every allocation seam call. */
function createCountingAllocatorDouble(): {
  allocator: NeatRacingRenderLoopAllocator;
  counter: { allocationCount: number };
} {
  const counter = { allocationCount: 0 };
  const allocator: NeatRacingRenderLoopAllocator = {
    float64Array(length: number): Float64Array {
      counter.allocationCount += 1;
      return new Float64Array(length);
    },
  };
  return { allocator, counter };
}

/**
 * Drives a fresh state through one simulated second of ideal frames at the
 * given display cadence and returns the total number of simulation steps.
 * Frame timestamps use `i * (1000 / hz)` exactly like AC-003's fixture
 * (16.67 / 8.33 / 6.94 ms deltas).
 */
function collectStepsPerSimulatedSecond(hz: number): number {
  const state = createNeatRacingRenderLoopState();
  let stepTotal = 0;
  for (let frameIndex = 0; frameIndex <= hz; frameIndex++) {
    stepTotal += advanceFrameSteps(state, frameIndex * (1000 / hz));
  }
  return stepTotal;
}

describe('neatRacing render-loop.utils (P1-S1-D)', () => {
  describe('cadence constants (spec sections 5.1/6.3)', () => {
    it('exposes the spec cadence constants', () => {
      expect(NEATRACING_FIXED_TIMESTEP_MS).toBeCloseTo(1000 / 60, 12);
      expect(NEATRACING_FIXED_TIMESTEP_SECONDS).toBeCloseTo(1 / 60, 12);
      expect(NEATRACING_MAX_CATCHUP_STEPS_PER_FRAME).toBe(4);
    });
  });

  describe('computeDeltaMs (recycled Neatenstein delta clamp, spec section 6.3)', () => {
    it('returns 0 for the first frame when no previous timestamp exists', () => {
      expect(computeDeltaMs(null, 100, 100)).toBe(0);
    });

    it('clamps an oversized tab-switch delta to the max delta', () => {
      expect(computeDeltaMs(0, 5000, 100)).toBe(100);
    });

    it('passes a normal frame delta through unclamped', () => {
      expect(computeDeltaMs(0, 16.6, 100)).toBe(16.6);
    });
  });

  describe('advanceFrameSteps accumulator (AC-003)', () => {
    it('runs zero steps on the very first frame (no previous timestamp)', () => {
      const state = createNeatRacingRenderLoopState();
      expect(advanceFrameSteps(state, NEATRACING_FIXED_TIMESTEP_MS)).toBe(0);
    });

    it('runs zero steps for a sub-timestep frame (never forced-min-1)', () => {
      const state = createNeatRacingRenderLoopState();
      const stepCount = advanceFrameSteps(
        state,
        NEATRACING_FIXED_TIMESTEP_MS / 2,
      );
      expect(stepCount).toBe(0);
    });

    it('carries sub-timestep remainders across frames until one step is due', () => {
      const state = createNeatRacingRenderLoopState();
      advanceFrameSteps(state, 0);
      const firstFrameSteps = advanceFrameSteps(state, 1000 / 120);
      const secondFrameSteps = advanceFrameSteps(state, 1000 / 60);
      expect(firstFrameSteps).toBe(0);
      expect(secondFrameSteps).toBe(1);
    });

    it('runs exactly 60 ticks over one simulated second at 60 Hz deltas (AC-003)', () => {
      expect(collectStepsPerSimulatedSecond(60)).toBe(60);
    });

    it('runs exactly 60 ticks over one simulated second at 120 Hz deltas (AC-003)', () => {
      expect(collectStepsPerSimulatedSecond(120)).toBe(60);
    });

    it('runs exactly 60 ticks over one simulated second at 144 Hz deltas (AC-003)', () => {
      expect(collectStepsPerSimulatedSecond(144)).toBe(60);
    });

    it('caps catch-up at the maximum steps per frame when a frame stalls', () => {
      const state = createNeatRacingRenderLoopState();
      advanceFrameSteps(state, 0);
      const stallStepCount = advanceFrameSteps(state, 100);
      expect(stallStepCount).toBe(NEATRACING_MAX_CATCHUP_STEPS_PER_FRAME);
    });

    it('drains the backlog after a stall instead of spiraling', () => {
      const state = createNeatRacingRenderLoopState();
      advanceFrameSteps(state, 0);
      advanceFrameSteps(state, 100);
      expect(state.accumulatedMs).toBeGreaterThanOrEqual(0);
      expect(state.accumulatedMs).toBeLessThan(NEATRACING_FIXED_TIMESTEP_MS);
    });
  });

  describe('zero per-frame allocations (AC-042 render-loop share, counting allocator double)', () => {
    it('allocates preallocated buffers only during state creation (AC-042)', () => {
      const { allocator, counter } = createCountingAllocatorDouble();
      const state = createNeatRacingRenderLoopState(allocator);
      const baselineAllocationCount = counter.allocationCount;
      for (let frameIndex = 0; frameIndex < 240; frameIndex++) {
        advanceFrameSteps(state, frameIndex * (1000 / 60));
      }
      expect(counter.allocationCount).toBe(baselineAllocationCount);
    });

    it('reuses the same preallocated matrix instances across frames', () => {
      const state = createNeatRacingRenderLoopState();
      const projectionBefore = state.projectionMatrix;
      const viewBefore = state.viewMatrix;
      advanceFrameSteps(state, NEATRACING_FIXED_TIMESTEP_MS);
      expect(state.projectionMatrix).toBe(projectionBefore);
      expect(state.viewMatrix).toBe(viewBefore);
    });

    it('preallocates 16-float matrices for projection and view', () => {
      const state = createNeatRacingRenderLoopState();
      expect(state.projectionMatrix.length).toBe(
        NEATRACING_RENDER_LOOP_MATRIX_FLOATS,
      );
      expect(state.viewMatrix.length).toBe(
        NEATRACING_RENDER_LOOP_MATRIX_FLOATS,
      );
    });

    it('preallocates 3-float vectors for camera position and forward axes', () => {
      const state = createNeatRacingRenderLoopState();
      expect(state.cameraPosition.length).toBe(
        NEATRACING_RENDER_LOOP_VECTOR_FLOATS,
      );
      expect(state.cameraForward.length).toBe(
        NEATRACING_RENDER_LOOP_VECTOR_FLOATS,
      );
    });
  });
});
