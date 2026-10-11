/**
 * Red-first contract tests for the NeatRacing editor save→rebuild pipeline
 * (slice P1-S6-E; spec section 8.3 policy P5 / AC-041).
 *
 * Locks the rebuild pipeline contract: edits during a drag first emit an
 * immediate dirty-slice remesh (the edited z-slice AABB ± 1 slices), then
 * schedule a trailing full rebuild debounced by 250 ms. Only one rebuild is
 * ever in flight; saves that arrive while a rebuild runs are coalesced so the
 * latest requested slice wins. A pointer-up flushes the debounce and runs the
 * full rebuild immediately. The pipeline tracks rebuild durations and
 * reports a p95 that stays within the 300 ms named budget on a fixture model.
 *
 * Snapshot contract: `inFlight` is true while the single in-flight rebuild
 * runs; `queued` is true while an accepted request waits behind that
 * rebuild; `supersededCount` counts saves that arrived while a rebuild was
 * in flight (the superseded rebuild's result is discarded, spec P5
 * latest-wins); `durationsMs` records only full-rebuild durations measured
 * through the injected `nowMs`; `p95Ms` is the 95th percentile of
 * `durationsMs`.
 *
 * The module is pure scheduling/state — no DOM, no WebGL, and no RNG — so
 * every contract is exercised with `jest.useFakeTimers()` in the plain Node
 * Jest environment. Rebuild work is stubbed deterministically: scheduling
 * contracts resolve instantly, single-flight contracts park on explicit
 * deferreds, and the p95 fixture measures durations through an injected
 * manual clock (`nowMs`), never an ambient wall clock.
 */
import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  jest,
} from '@jest/globals';
import {
  createEditorRebuildPipeline,
  NEATRACING_REBUILD_DEBOUNCE_MS,
  NEATRACING_REBUILD_P95_BUDGET_MS,
} from './editor.rebuild.pipeline';
import type { EditorRebuildRequest } from './editor.rebuild.pipeline';

/**
 * Instant rebuild double for the scheduling contracts: every request resolves
 * without touching the timer queue, so debounce scheduling is the only thing
 * the fake timers drive in those contracts.
 */
function makeInstantRebuild(): jest.Mock<
  (request: EditorRebuildRequest) => Promise<void>
> {
  return jest.fn(async (_request: EditorRebuildRequest) => {
    // Instant double: resolves without handling the payload; consuming the
    // request reference keeps the declared executor arity lint-clean.
    void _request;
  });
}

/**
 * Manual monotonic clock double for the p95 fixture. The pipeline must time
 * its rebuild durations through the injected `nowMs` accessor, never an
 * ambient wall clock, so the fixture controls every recorded duration.
 */
function makeManualClock(): {
  nowMs: () => number;
  advance: (ms: number) => void;
} {
  let now = 1000;
  return {
    nowMs: () => now,
    advance: (ms: number) => {
      now += ms;
    },
  };
}

/**
 * Controlled rebuild double for the single-flight contracts: dirty remeshes
 * resolve instantly, full rebuilds park on a deferred promise the test
 * resolves explicitly, and the harness records the peak concurrent in-flight
 * count to pin the one-rebuild-at-a-time guarantee.
 */
function makeControlledFullRebuild(): {
  rebuild: jest.Mock<(request: EditorRebuildRequest) => Promise<void>>;
  settleFirstFull: () => void;
  maxInFlight: () => number;
} {
  const parkedFullResolvers: Array<() => void> = [];
  let inFlight = 0;
  let maxInFlight = 0;
  const rebuild = jest.fn(async (request: EditorRebuildRequest) => {
    if (request.mode !== 'full') {
      return;
    }
    inFlight += 1;
    maxInFlight = Math.max(maxInFlight, inFlight);
    await new Promise<void>((resolve) =>
      parkedFullResolvers.push(resolve),
    ).then(() => {
      inFlight -= 1;
    });
  });
  return {
    rebuild,
    settleFirstFull: () => {
      const resolve = parkedFullResolvers.shift();
      if (!resolve) {
        throw new Error('settleFirstFull: no parked full rebuild in flight');
      }
      resolve();
    },
    maxInFlight: () => maxInFlight,
  };
}

/**
 * Advances fake timers by the requested number of milliseconds and then
 * flushes the microtask queue so any promises resolved by the timer ticks
 * have settled before the next assertion.
 */
async function advanceMs(ms: number): Promise<void> {
  await jest.advanceTimersByTimeAsync(ms);
  await Promise.resolve();
}

describe('editor.rebuild.pipeline (P1-S6-E; spec AC-041)', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  describe('named timing constants', () => {
    it('exports a 250 ms trailing debounce constant', () => {
      expect(NEATRACING_REBUILD_DEBOUNCE_MS).toBe(250);
    });

    it('exports a 300 ms p95 budget constant', () => {
      expect(NEATRACING_REBUILD_P95_BUDGET_MS).toBe(300);
    });
  });

  describe('dirty-slice remesh during drag', () => {
    it('emits an immediate dirty remesh covering the edited slice AABB ± 1', async () => {
      const rebuild = makeInstantRebuild();
      const pipeline = createEditorRebuildPipeline({ rebuild });
      pipeline.edit(42);
      await advanceMs(0);
      expect(rebuild).toHaveBeenCalledWith({
        mode: 'dirty',
        slices: [41, 42, 43],
      });
    });

    it('clamps the dirty window at the volume edges, deduplicating the pinned slice', async () => {
      const rebuild = makeInstantRebuild();
      const pipeline = createEditorRebuildPipeline({ rebuild });
      pipeline.edit(0);
      pipeline.edit(127);
      await advanceMs(0);
      expect(rebuild).toHaveBeenCalledTimes(2);
      expect(rebuild).toHaveBeenNthCalledWith(1, {
        mode: 'dirty',
        slices: [0, 1],
      });
      expect(rebuild).toHaveBeenNthCalledWith(2, {
        mode: 'dirty',
        slices: [126, 127],
      });
    });
  });

  describe('250 ms trailing debounce on full rebuild', () => {
    it('does not run a full rebuild before the debounce window closes', async () => {
      const rebuild = makeInstantRebuild();
      const pipeline = createEditorRebuildPipeline({ rebuild });
      pipeline.edit(5);
      await advanceMs(NEATRACING_REBUILD_DEBOUNCE_MS - 1);
      const fullCalls = rebuild.mock.calls.filter(
        (call) => call[0].mode === 'full',
      );
      expect(fullCalls).toHaveLength(0);
    });

    it('runs exactly one coalesced full rebuild when the debounce window closes', async () => {
      const rebuild = makeInstantRebuild();
      const pipeline = createEditorRebuildPipeline({ rebuild });
      pipeline.edit(5);
      pipeline.edit(6);
      await advanceMs(NEATRACING_REBUILD_DEBOUNCE_MS);
      const fullCalls = rebuild.mock.calls.filter(
        (call) => call[0].mode === 'full',
      );
      expect(fullCalls).toHaveLength(1);
      expect(fullCalls[0]).toEqual([{ mode: 'full', slices: [] }]);
    });
  });

  describe('latest-wins single-flight policy', () => {
    it('keeps at most one rebuild in flight and queues the mid-flight save behind it', async () => {
      const controlled = makeControlledFullRebuild();
      const pipeline = createEditorRebuildPipeline({
        rebuild: controlled.rebuild,
      });
      pipeline.edit(1);
      await advanceMs(NEATRACING_REBUILD_DEBOUNCE_MS);
      expect(controlled.rebuild).toHaveBeenCalledTimes(2);
      const callsWhileIdle = controlled.rebuild.mock.calls.length;

      pipeline.edit(2);
      await advanceMs(0);
      expect(controlled.rebuild.mock.calls).toHaveLength(callsWhileIdle);

      controlled.settleFirstFull();
      await advanceMs(0);
      expect(controlled.maxInFlight()).toBe(1);
    });

    it('reports the in-flight and queued state through snapshot', async () => {
      const controlled = makeControlledFullRebuild();
      const pipeline = createEditorRebuildPipeline({
        rebuild: controlled.rebuild,
      });
      pipeline.edit(1);
      await advanceMs(NEATRACING_REBUILD_DEBOUNCE_MS);

      pipeline.edit(2);
      await advanceMs(0);
      const midFlight = pipeline.snapshot();
      expect(midFlight.inFlight).toBe(true);
      expect(midFlight.queued).toBe(true);
    });

    it('counts a save arriving mid-flight as one superseded rebuild result', async () => {
      const controlled = makeControlledFullRebuild();
      const pipeline = createEditorRebuildPipeline({
        rebuild: controlled.rebuild,
      });
      pipeline.edit(1);
      await advanceMs(NEATRACING_REBUILD_DEBOUNCE_MS);
      pipeline.edit(5);
      await advanceMs(0);
      expect(pipeline.snapshot().supersededCount).toBe(1);
    });
  });

  describe('pointer-up flush', () => {
    it('flushes one immediate full rebuild on pointer-up and consumes the pending debounce', async () => {
      const rebuild = makeInstantRebuild();
      const pipeline = createEditorRebuildPipeline({ rebuild });
      pipeline.edit(7);
      await advanceMs(0);
      pipeline.pointerUp();
      await advanceMs(0);
      const fullCalls = rebuild.mock.calls.filter(
        (call) => call[0].mode === 'full',
      );
      expect(fullCalls).toHaveLength(1);
      expect(fullCalls[0]).toEqual([{ mode: 'full', slices: [] }]);
      await advanceMs(NEATRACING_REBUILD_DEBOUNCE_MS);
      expect(
        rebuild.mock.calls.filter((call) => call[0].mode === 'full'),
      ).toHaveLength(1);
    });
  });

  describe('p95 budget', () => {
    it('reports p95 within the 300 ms budget on the 19x290 + 1x350 injected-clock fixture', async () => {
      const clock = makeManualClock();
      const fullDurations = [
        290, 290, 290, 290, 290, 290, 290, 290, 290, 290, 290, 290, 290, 290,
        290, 290, 290, 290, 290, 350,
      ];
      let fullIndex = 0;
      const rebuild = jest.fn(async (request: EditorRebuildRequest) => {
        if (request.mode !== 'full') {
          return;
        }
        clock.advance(fullDurations[fullIndex]);
        fullIndex += 1;
      });
      const pipeline = createEditorRebuildPipeline({
        rebuild,
        nowMs: clock.nowMs,
      });

      for (let i = 0; i < fullDurations.length; i += 1) {
        pipeline.edit(i);
        await advanceMs(NEATRACING_REBUILD_DEBOUNCE_MS);
      }

      const stats = pipeline.snapshot();
      expect(stats.durationsMs).toHaveLength(20);
      expect(stats.p95Ms).toBeGreaterThanOrEqual(285);
      expect(stats.p95Ms).toBeLessThanOrEqual(NEATRACING_REBUILD_P95_BUDGET_MS);
    });
  });
});
