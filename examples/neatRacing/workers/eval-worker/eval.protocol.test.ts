import { afterAll, beforeAll, describe, expect, it } from '@jest/globals';
import {
  NEATRACING_VARIANT_STREAM_SURFACE_TAG,
  combineNeatRacingChildSeed,
  createNeatRacingXorShift32Stream,
  deriveNeatRacingVariantStreamChildSeed,
} from '../simulation-worker/reproducibility.utils';
import {
  NEATRACING_EVAL_WORKER_PARALLEL_VARIANT_COUNT,
  runBoundedConcurrency,
} from './eval.protocol';
import type {
  NeatRacingEvalVariantUnit,
} from './eval-worker';
import type {
  NeatRacingEvalWorkerRequest,
  NeatRacingEvalWorkerResponse,
  NeatRacingEvalWorkerResultPayload,
  NeatRacingEvalWorkerVariant,
} from './eval.protocol';

/**
 * Green tests for the neatRacing dedicated eval worker and its sim ↔ eval
 * message protocol (Phase 2, Step 02, slice P2-S2-A — tdd_sequence
 * `green-only`).
 *
 * The red-test note for this slice is superseded by the green-only
 * sequence: there is no RED phase, and this suite asserts the slice's
 * acceptance surface — the request/eval/result protocol round trip with
 * the `(seed, requestTick)` echo (E11(c)), the bounded live-mode
 * parallelVariantCount named constant instead of the curriculum demo's
 * 256 (TP-18), round-boundary `stop` cancellation that lets no
 * superseded-round reply escape (E11(d)), and the §9.7.2 row-4 per-variant
 * seed derivation the eval worker owns (D2(b)).
 *
 * Testing pattern: the flappy worker precedent — `globalThis.self` is
 * replaced with a fake worker scope BEFORE the worker module is dynamically
 * imported (TS's CommonJS emit hoists static requires, so only a dynamic
 * import runs after the fake scope binds; the module attaches its message
 * adapter at import time in a worker realm). The fake is restored in
 * `afterAll`, and every behavioral assertion drives the pure
 * `NEATRACING_EVAL_WORKER_INTERNALS` session state machine without a real
 * worker realm.
 */

/** Lazily-loaded worker module type (import in `beforeAll` after the stub). */
type EvalWorkerModule = typeof import('./eval-worker');

/** Session handle type alias for helper signatures. */
type EvalWorkerSession = ReturnType<
  EvalWorkerModule['NEATRACING_EVAL_WORKER_INTERNALS']['createEvalWorkerSession']
>;

/** Fake DedicatedWorkerGlobalScope the module's onmessage adapter binds to. */
const evalWorkerScope: { onmessage: unknown; postMessage: jest.Mock } = {
  onmessage: undefined,
  postMessage: jest.fn(),
};
/** Original `self` property descriptor captured before the test fakes it. */
const originalSelfDescriptor = Object.getOwnPropertyDescriptor(
  globalThis,
  'self',
);

Object.defineProperty(globalThis, 'self', {
  configurable: true,
  value: evalWorkerScope,
});

afterAll(() => {
  if (originalSelfDescriptor === undefined) {
    Object.defineProperty(globalThis, 'self', {
      configurable: true,
      value: undefined,
    });
  } else {
    Object.defineProperty(globalThis, 'self', originalSelfDescriptor);
  }
});

/** Worker module loaded in `beforeAll` once the fake scope has bound. */
let evalWorkerModule: EvalWorkerModule;

beforeAll(async () => {
  evalWorkerModule = await import('./eval-worker');
});

/** Fixture AC-026 tuple seed for the eval protocol tests. */
const EVAL_TEST_SEED = 4242;

/** Fixture request tick (a deterministic adaptation boundary). */
const EVAL_TEST_REQUEST_TICK = 96;

/**
 * Builds an eval request payload fixture: seed, request tick, and a
 * variant batch (defaulting to one well-formed slot).
 *
 * @param overrides - Per-test overrides; unset fields fall back to fixtures.
 * @returns A well-formed eval request payload.
 */
function createEvalTestFixturePayload(overrides: {
  seed?: number;
  requestTick?: number;
  variants?: readonly NeatRacingEvalWorkerVariant[];
} = {}): NeatRacingEvalWorkerEvalPayloadFixture {
  return {
    seed: overrides.seed ?? EVAL_TEST_SEED,
    requestTick: overrides.requestTick ?? EVAL_TEST_REQUEST_TICK,
    variants:
      overrides.variants ?? [{ variantIndex: 0, snapshot: { kind: 'fixture' } }],
  };
}

/**
 * Local structural alias so the fixture builder stays decoupled from
 * runtime imports: identical shape to the worker module's eval payload.
 */
type NeatRacingEvalWorkerEvalPayloadFixture = {
  seed: number;
  requestTick: number;
  variants: readonly NeatRacingEvalWorkerVariant[];
};

/**
 * Builds a typed eval request for the session state machine.
 *
 * @param payload - The eval request payload fixture.
 * @returns The `eval` request union member.
 */
function createEvalTestFixtureRequest(
  payload: NeatRacingEvalWorkerEvalPayloadFixture,
): NeatRacingEvalWorkerRequest {
  return { type: 'eval', payload };
}

/**
 * Builds a contiguous variant batch `0..count - 1` with fixture snapshots.
 *
 * @param count - Number of variant slots.
 * @param patchSize - Optional shared patch size for the stride table.
 * @returns The variant batch.
 */
function createContiguousVariantBatch(
  count: number,
  patchSize?: number,
): NeatRacingEvalWorkerVariant[] {
  return Array.from({ length: count }, (_, variantIndex) => ({
    variantIndex,
    ...(patchSize === undefined ? {} : { patchSize }),
    snapshot: { kind: 'fixture', variantIndex },
  }));
}

/**
 * A variant unit whose every evaluation blocks until explicitly released,
 * recording its entry order — the harness for concurrency-bound,
 * completion-order, and cancellation assertions.
 */
interface BlockedEvalUnitHarness {
  /** Variant indexes in the order their evaluations were entered. */
  enteredIndexes: number[];
  /** Releases the evaluation entered for one variant index. */
  release: (variantIndex: number) => void;
  /** Releases every entered evaluation captured so far. */
  releaseAll: () => void;
  /** The unit to bind into a session. */
  unit: NeatRacingEvalVariantUnit;
}

/**
 * Creates the blocked-evaluation harness. Every entered evaluation pushes
 * its release and resolves to the slot's variantIndex, so released results
 * double as an order-preserving fingerprint.
 *
 * @returns The harness for driving bounded concurrency assertions.
 */
function createBlockedEvalUnitHarness(): BlockedEvalUnitHarness {
  const enteredIndexes: number[] = [];
  const resolvers: Array<() => void> = [];
  const unit: NeatRacingEvalVariantUnit = ({ variant }) => {
    enteredIndexes.push(variant.variantIndex);
    return new Promise<number>((resolve) => {
      resolvers.push(() => resolve(variant.variantIndex));
    });
  };
  return {
    enteredIndexes,
    release: (variantIndex) => {
      const entryPosition = enteredIndexes.indexOf(variantIndex);
      resolvers[entryPosition]();
    },
    releaseAll: () => {
      for (const resolve of [...resolvers]) {
        resolve();
      }
    },
    unit,
  };
}

/**
 * A unit that answers immediately with a deterministic score derived only
 * from the slot's payload — the happy-path fingerprint (score =
 * variantIndex + 1).
 */
const immediateUnit: NeatRacingEvalVariantUnit = ({ variant }) =>
  Promise.resolve(variant.variantIndex + 1);

/**
 * Creates a session through the INTERNALS seam with a bound unit.
 *
 * @param evaluateVariant - The unit this session evaluates with.
 * @returns The fresh eval worker session.
 */
function createTestEvalWorkerSession(
  evaluateVariant: NeatRacingEvalVariantUnit,
): EvalWorkerSession {
  return evalWorkerModule.NEATRACING_EVAL_WORKER_INTERNALS.createEvalWorkerSession(
    { evaluateVariant },
  );
}

/** Drains the promise microtask queue so bounded slots settle. */
async function flushMicrotasks(rounds = 24): Promise<void> {
  for (let round = 0; round < rounds; round += 1) {
    await Promise.resolve();
  }
}

describe('NEATRACING_EVAL_WORKER_INTERNALS', () => {
  describe('protocol contract (eval → result)', () => {
    it('answers one eval request with a single result reply echoing seed, requestTick, and ordered scores', async () => {
      const session = createTestEvalWorkerSession(immediateUnit);
      const request = createEvalTestFixtureRequest(
        createEvalTestFixturePayload({
          variants: createContiguousVariantBatch(3),
        }),
      );

      const responses = await session.handleRequest(request);

      expect(responses).toEqual([
        {
          type: 'result',
          payload: {
            seed: EVAL_TEST_SEED,
            requestTick: EVAL_TEST_REQUEST_TICK,
            results: [
              { variantIndex: 0, score: 1 },
              { variantIndex: 1, score: 2 },
              { variantIndex: 2, score: 3 },
            ],
          },
        },
      ]);
    });

    it('keeps identical (seed, variants) requests replayable across sessions', async () => {
      const request = createEvalTestFixtureRequest(
        createEvalTestFixturePayload({
          variants: createContiguousVariantBatch(4),
        }),
      );

      const first = await createTestEvalWorkerSession(
        evalWorkerModule.NEATRACING_EVAL_WORKER_INTERNALS.createNeatRacingBootstrapVariantUnit(),
      ).handleRequest(request);
      const second = await createTestEvalWorkerSession(
        evalWorkerModule.NEATRACING_EVAL_WORKER_INTERNALS.createNeatRacingBootstrapVariantUnit(),
      ).handleRequest(request);

      expect(first).toEqual(second);
    });

    it('derives per-variant streams from the §9.7.2 row-4 stride table, including the patchSize stride widening', async () => {
      const session = createTestEvalWorkerSession(
        evalWorkerModule.NEATRACING_EVAL_WORKER_INTERNALS.createNeatRacingBootstrapVariantUnit(),
      );
      const variants: NeatRacingEvalWorkerVariant[] = [
        { variantIndex: 0, snapshot: 'a' },
        { variantIndex: 2, patchSize: 500, snapshot: 'b' },
        { variantIndex: 5, snapshot: 'c' },
      ];
      const request = createEvalTestFixtureRequest(
        createEvalTestFixturePayload({ variants }),
      );

      const responses = await session.handleRequest(request);

      // The bootstrap unit scores each slot with the FIRST draw of its
      // derived variant stream, so expected scores re-play the row-4 table
      // by hand: streamBase = mix(tupleSeed, 'variant'); slot seed =
      // streamBase + variantIndex × stride; stride = max(offset base,
      // patchSize × stride factor).
      const streamBase = combineNeatRacingChildSeed(
        EVAL_TEST_SEED,
        NEATRACING_VARIANT_STREAM_SURFACE_TAG,
      );
      const patchStridedSlotSeed =
        deriveNeatRacingVariantStreamChildSeed(EVAL_TEST_SEED, 2, 500);
      expect(patchStridedSlotSeed).toBe((streamBase + 2 * 2000) >>> 0);
      // The patchSize widened the stride above the offset base...
      expect(patchStridedSlotSeed).not.toBe((streamBase + 2 * 1000) >>> 0);
      // ...while slots without a patch size ride the recycled base stride.
      expect(deriveNeatRacingVariantStreamChildSeed(EVAL_TEST_SEED, 5, 0)).toBe(
        (streamBase + 5 * 1000) >>> 0,
      );

      expect(responses[0].type).toBe('result');
      const payload = (responses[0] as { payload: NeatRacingEvalWorkerResultPayload })
        .payload;
      const expectedScores = variants.map((variant) =>
        createNeatRacingXorShift32Stream(
          deriveNeatRacingVariantStreamChildSeed(
            EVAL_TEST_SEED,
            variant.variantIndex,
            variant.patchSize ?? 0,
          ),
        )(),
      );
      expectedScores.forEach((expectedScore, index) => {
        expect(payload.results[index].variantIndex).toBe(
          variants[index].variantIndex,
        );
        expect(payload.results[index].score).toBe(expectedScore);
      });
    });
  });

  describe('bounded parallelVariantCount (TP-18)', () => {
    it('exposes the named live-mode bound below the curriculum demo window', () => {
      expect(NEATRACING_EVAL_WORKER_PARALLEL_VARIANT_COUNT).toBe(16);
      expect(NEATRACING_EVAL_WORKER_PARALLEL_VARIANT_COUNT).toBeLessThan(256);
    });

    it('dispatches at most the named constant concurrent evaluations across an oversized batch', async () => {
      const harness = createBlockedEvalUnitHarness();
      const session = createTestEvalWorkerSession(harness.unit);
      const batchCount = NEATRACING_EVAL_WORKER_PARALLEL_VARIANT_COUNT + 4;
      const request = createEvalTestFixtureRequest(
        createEvalTestFixturePayload({
          variants: createContiguousVariantBatch(batchCount),
        }),
      );

      const requestPromise = session.handleRequest(request);
      await flushMicrotasks();

      // Only one bounded window is live, however large the batch is.
      expect(harness.enteredIndexes).toHaveLength(
        NEATRACING_EVAL_WORKER_PARALLEL_VARIANT_COUNT,
      );

      // Releasing the window opens exactly the remaining slots.
      harness.releaseAll();
      await flushMicrotasks();
      harness.releaseAll();
      await flushMicrotasks();
      expect(harness.enteredIndexes).toHaveLength(batchCount);

      const responses = await requestPromise;
      expect(responses).toHaveLength(1);
      expect(responses[0].type).toBe('result');
      const payload = (responses[0] as { payload: NeatRacingEvalWorkerResultPayload })
        .payload;
      expect(payload.results).toHaveLength(batchCount);
      payload.results.forEach((result, index) => {
        expect(result.variantIndex).toBe(index);
        expect(result.score).toBe(index);
      });
    });

    it('reports results in request order regardless of completion order', async () => {
      const harness = createBlockedEvalUnitHarness();
      const session = createTestEvalWorkerSession(harness.unit);
      const request = createEvalTestFixtureRequest(
        createEvalTestFixturePayload({
          variants: createContiguousVariantBatch(4),
        }),
      );

      const requestPromise = session.handleRequest(request);
      await flushMicrotasks();
      expect(harness.enteredIndexes).toEqual([0, 1, 2, 3]);

      // Complete slot 3 first and slot 0 last: reporting order still rides
      // the request order, never the completion order (D2 stability).
      harness.release(3);
      await flushMicrotasks(8);
      harness.release(2);
      await flushMicrotasks(8);
      harness.release(1);
      await flushMicrotasks(8);
      harness.release(0);
      await flushMicrotasks(8);

      const responses = await requestPromise;
      expect(responses).toHaveLength(1);
      expect(responses[0].type).toBe('result');
      const payload = (responses[0] as { payload: NeatRacingEvalWorkerResultPayload })
        .payload;
      expect(
        payload.results.map((result) => result.variantIndex),
      ).toEqual([0, 1, 2, 3]);
      expect(payload.results.map((result) => result.score)).toEqual([
        0, 1, 2, 3,
      ]);
    });
  });

  describe('round-boundary cancellation (E11(d))', () => {
    it('stop discards the in-flight window: no reply escapes and no further slots open', async () => {
      const harness = createBlockedEvalUnitHarness();
      const session = createTestEvalWorkerSession(harness.unit);
      const batchCount = NEATRACING_EVAL_WORKER_PARALLEL_VARIANT_COUNT + 4;
      const request = createEvalTestFixtureRequest(
        createEvalTestFixturePayload({
          variants: createContiguousVariantBatch(batchCount),
        }),
      );

      const requestPromise = session.handleRequest(request);
      await flushMicrotasks();
      expect(harness.enteredIndexes).toHaveLength(
        NEATRACING_EVAL_WORKER_PARALLEL_VARIANT_COUNT,
      );

      // Round boundary: the stop flips the phase and posts nothing.
      expect(await session.handleRequest({ type: 'stop' })).toEqual([]);
      expect(session.snapshot().phase).toBe('stopped');

      // Releasing the in-flight units discards the window wholesale: the
      // request resolves with NO reply, and the four pending slots never
      // open (no superseded-round evaluation runs after the boundary).
      harness.releaseAll();
      await flushMicrotasks();
      expect(harness.enteredIndexes).toHaveLength(
        NEATRACING_EVAL_WORKER_PARALLEL_VARIANT_COUNT,
      );
      expect(await requestPromise).toEqual([]);
    });

    it('rejects eval requests after a stop with an error reply naming the round boundary', async () => {
      const session = createTestEvalWorkerSession(immediateUnit);
      await session.handleRequest({ type: 'stop' });

      const responses = await session.handleRequest(
        createEvalTestFixtureRequest(createEvalTestFixturePayload()),
      );

      expect(responses).toHaveLength(1);
      expect(responses[0].type).toBe('error');
      const message = (responses[0] as { payload: { message: string } }).payload
        .message;
      expect(message).toContain('stopped');
    });

    it('tolerates stops with no in-flight work and stays idempotent', async () => {
      const session = createTestEvalWorkerSession(immediateUnit);

      expect(await session.handleRequest({ type: 'stop' })).toEqual([]);
      expect(await session.handleRequest({ type: 'stop' })).toEqual([]);
      expect(session.snapshot().phase).toBe('stopped');
    });
  });

  describe('malformed-message contract', () => {
    const malformedCases: Array<{ name: string; request: unknown }> = [
      { name: 'a null message', request: null },
      { name: 'an unknown message type', request: { type: 'nope' } },
      { name: 'a scalar eval payload', request: { type: 'eval', payload: 7 } },
      {
        name: 'a non-finite seed',
        request: {
          type: 'eval',
          payload: createEvalTestFixturePayload({ seed: Number.NaN }),
        },
      },
      {
        name: 'a negative requestTick',
        request: {
          type: 'eval',
          payload: createEvalTestFixturePayload({ requestTick: -1 }),
        },
      },
      {
        name: 'a fractional requestTick',
        request: {
          type: 'eval',
          payload: createEvalTestFixturePayload({ requestTick: 1.5 }),
        },
      },
      {
        name: 'an empty variant batch',
        request: {
          type: 'eval',
          payload: createEvalTestFixturePayload({ variants: [] }),
        },
      },
      {
        name: 'a negative variantIndex',
        request: {
          type: 'eval',
          payload: createEvalTestFixturePayload({
            variants: [{ variantIndex: -1, snapshot: 1 }],
          }),
        },
      },
      {
        name: 'a negative patchSize',
        request: {
          type: 'eval',
          payload: createEvalTestFixturePayload({
            variants: [{ variantIndex: 0, patchSize: -1, snapshot: 1 }],
          }),
        },
      },
      {
        name: 'a missing snapshot',
        request: {
          type: 'eval',
          payload: createEvalTestFixturePayload({
            variants: [{ variantIndex: 0 } as NeatRacingEvalWorkerVariant],
          }),
        },
      },
    ];

    malformedCases.forEach((item) => {
      it(`rejects ${item.name} with a single error reply`, async () => {
        const session = createTestEvalWorkerSession(immediateUnit);

        const responses = await session.handleRequest(
          item.request as NeatRacingEvalWorkerRequest,
        );

        expect(responses).toEqual([
          { type: 'error', payload: { message: expect.any(String) } },
        ]);
      });
    });
  });

  describe('runBoundedConcurrency harness', () => {
    it('rejects non-positive and fractional dispatch bounds with a RangeError', async () => {
      await expect(runBoundedConcurrency([], 0)).rejects.toThrow(RangeError);
      await expect(runBoundedConcurrency([], -1)).rejects.toThrow(RangeError);
      await expect(runBoundedConcurrency([], 1.5)).rejects.toThrow(RangeError);
    });

    it('resolves an empty task list without opening any slot', async () => {
      const results = await runBoundedConcurrency<number>(
        [],
        NEATRACING_EVAL_WORKER_PARALLEL_VARIANT_COUNT,
      );

      expect(results).toEqual([]);
    });
  });

  describe('worker-transport adapter', () => {
    it('wired onmessage posts one result per eval, nothing per stop, one error per junk', async () => {
      expect(typeof evalWorkerScope.onmessage).toBe('function');
      evalWorkerScope.postMessage.mockClear();

      const handler = evalWorkerScope.onmessage as (event: {
        data: unknown;
      }) => void;

      // Act: one real eval window (module session → bootstrap unit), one
      // round-boundary stop, one junk message.
      const variant = { variantIndex: 0, snapshot: 'fixture' };
      handler({
        data: createEvalTestFixtureRequest(
          createEvalTestFixturePayload({ variants: [variant] }),
        ),
      });
      await flushMicrotasks(8);
      expect(evalWorkerScope.postMessage.mock.calls).toHaveLength(1);

      handler({ data: { type: 'stop' } });
      await flushMicrotasks(8);
      // Stop posts nothing (fire-and-forget lifecycle message).
      expect(evalWorkerScope.postMessage.mock.calls).toHaveLength(1);

      handler({ data: { type: 'nonsense' } });
      await flushMicrotasks(8);
      expect(evalWorkerScope.postMessage.mock.calls).toHaveLength(2);

      // Assert: the module-level session answered on the fake worker scope.
      const posts = evalWorkerScope.postMessage.mock.calls.map(
        (call) => call[0] as NeatRacingEvalWorkerResponse,
      );
      expect(posts[0].type).toBe('result');
      const resultPayload = (posts[0] as { payload: NeatRacingEvalWorkerResultPayload })
        .payload;
      expect(resultPayload.seed).toBe(EVAL_TEST_SEED);
      expect(resultPayload.requestTick).toBe(EVAL_TEST_REQUEST_TICK);
      expect(resultPayload.results).toHaveLength(1);
      // The bootstrap unit scored slot 0 with the FIRST draw of its derived
      // variant stream — the full adapter → session → unit → derivation path
      // replayed by hand (D2(b) end to end).
      expect(resultPayload.results[0].score).toBe(
        createNeatRacingXorShift32Stream(
          deriveNeatRacingVariantStreamChildSeed(EVAL_TEST_SEED, 0, 0),
        )(),
      );
      expect(posts[1].type).toBe('error');
    });
  });
});
