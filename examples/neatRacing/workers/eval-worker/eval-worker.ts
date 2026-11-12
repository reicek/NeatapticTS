/// <reference lib="webworker" />
/**
 * neatRacing dedicated eval worker — growth/variant evaluation side of the
 * two-worker split (Phase 2, Step 02, slice P2-S2-A — tdd_sequence
 * `green-only`).
 *
 * Spec anchors:
 * - §9.2 file tree: "eval-worker.ts — NGE growth/variant evaluation (P2;
 *   neatenstein eval.worker.ts precedent)". The simulation worker stays
 *   60 Hz physics + per-tick control (E11(a)); everything growth/variant
 *   lives here (E11(b)).
 * - E11(c): eval replies are queued by the sim side and applied only at
 *   deterministic adaptation-tick boundaries. This worker therefore stamps
 *   every reply with the request's `(seed, requestTick)` echo instead of
 *   any wall-clock or in-flight state.
 * - E11(d): a `stop` is the round-boundary cancellation. It is terminal:
 *   in-flight windows are discarded (no reply escapes), pending slots
 *   refuse to open, and later eval requests answer with an `error` reply
 *   telling the caller to start a fresh worker for the new round.
 * - D2: every variant's RNG stream is derived inside this worker from the
 *   request's tuple seed (§9.7.2 row 4 via the recycled reproducibility
 *   table) — this worker never consumes or advances sim-worker network RNG
 *   state.
 * - TP-18: the bounded dispatch constant lives in ./eval.protocol.ts and
 *   drives runBoundedConcurrency below.
 *
 * House pattern (matches ../simulation-worker): a pure session state
 * machine factory whose `handleRequest` resolves the ordered replies to
 * post, an INTERNALS const export as the Jest seam, and a module-level
 * session + `self.onmessage` adapter as the only file impurity.
 */
import {
  createNeatRacingXorShift32Stream,
  deriveNeatRacingVariantStreamChildSeed,
} from '../simulation-worker/reproducibility.utils';
import {
  runBoundedConcurrency,
  validateNeatRacingEvalWorkerEvalPayload,
  type NeatRacingEvalWorkerEvalPayload,
  type NeatRacingEvalWorkerRequest,
  type NeatRacingEvalWorkerResponse,
  type NeatRacingEvalWorkerResultPayload,
  type NeatRacingEvalWorkerTask,
  type NeatRacingEvalWorkerVariant,
} from './eval.protocol';

/** Inputs the worker hands its variant-evaluation unit (D2 + §9.7.2). */
export interface NeatRacingEvalVariantUnitContext {
  /** The validated variant slot as received on the wire. */
  variant: NeatRacingEvalWorkerVariant;
  /**
   * §9.7.2 row-4 child seed for this slot —
   * deriveNeatRacingVariantStreamChildSeed(tupleSeed, variantIndex,
   * patchSize ?? 0). Owned by the eval worker (D2(b)).
   */
  variantSeed: number;
  /** The request's AC-026 tuple seed, for engine-side child derivations. */
  tupleSeed: number;
}

/**
 * A variant evaluation unit: produces one deterministic fitness score per
 * slot (sync or async). The Step 03 NGE growth engine plugs in here
 * through the session's `evaluateVariant` option without touching the
 * protocol or the bounded dispatch.
 */
export type NeatRacingEvalVariantUnit = (
  context: NeatRacingEvalVariantUnitContext,
) => number | Promise<number>;

/** Eval worker phases: ready until a round-boundary `stop` lands. */
export type NeatRacingEvalWorkerPhase = 'ready' | 'stopped';

/** Session snapshot for green tests (read-only view of the phase). */
export interface NeatRacingEvalWorkerSessionSnapshot {
  phase: NeatRacingEvalWorkerPhase;
}

/** Eval worker session contract (pure state machine, no worker realm). */
export interface NeatRacingEvalWorkerSession {
  /**
   * Routes one request and resolves the ordered replies to post. `eval`
   * resolves one result reply (or one error reply, or — after cancellation
   * — no reply at all); `stop` resolves no reply (fire-and-forget
   * lifecycle message, matching the flappy stop precedent).
   */
  handleRequest(
    request: NeatRacingEvalWorkerRequest,
  ): Promise<NeatRacingEvalWorkerSessionResponse[]>;
  /** Read-only phase view for tests. */
  snapshot(): NeatRacingEvalWorkerSessionSnapshot;
}

/**
 * The eval-side response union (aliasing the protocol wire type so the
 * session's signature reads as its own contract).
 */
export type NeatRacingEvalWorkerSessionResponse = NeatRacingEvalWorkerResponse;

/**
 * Private round-boundary cancellation sentinel thrown by in-flight task
 * wrappers after a `stop`. Never escapes the session: `handleEval`
 * converts it into "no reply" (E11(d)) rather than an error reply.
 */
class EvalWorkerCancelled extends Error {
  constructor() {
    super('neatracing eval worker: cancelled by round-boundary stop');
    this.name = 'EvalWorkerCancelled';
  }
}

/**
 * Builds the single error reply for contract-level rejections.
 *
 * @param message - Static rejection vocabulary.
 * @returns The error response to post.
 */
function createEvalWorkerErrorReply(
  message: string,
): NeatRacingEvalWorkerResponse {
  return { type: 'error', payload: { message } };
}

/**
 * Creates the protocol-bootstrap variant unit used until the Step 03 NGE
 * growth engine lands (the remaining AC-029/AC-043 surface).
 *
 * Educational note: the unit scores each slot from the variant stream
 * itself (D2(b)) — it draws the FIRST xorshift32 float of the stream
 * seeded with the per-variant child seed. The eval worker's end-to-end
 * behavior stays deterministic and replayable (same `(seed, variants)` →
 * identical scores) while still exercising the exact §9.7.2 derivation
 * table the real engine will receive. Step 03 swaps this unit through the
 * session's `evaluateVariant` option — the protocol, the seed derivation,
 * and the bounded dispatch all carry over unchanged.
 *
 * @returns The bootstrap evaluation unit.
 */
export function createNeatRacingBootstrapVariantUnit(): NeatRacingEvalVariantUnit {
  return ({ variantSeed }: NeatRacingEvalVariantUnitContext) =>
    createNeatRacingXorShift32Stream(variantSeed)();
}

/**
 * Creates a fresh eval-worker session (pure, no worker realm).
 *
 * Overlapping eval requests are legal: each window is independent and
 * self-contained (its own tuple seed + variants + echoed identity), which
 * is what lets a Step 03 sim-side queue pipeline windows without the eval
 * worker holding cross-window state — and keeps every reply replayable
 * from its request alone (D2).
 *
 * @param options - The variant evaluation unit bound to this session.
 * @returns The eval worker session state machine.
 */
export function createEvalWorkerSession(options: {
  evaluateVariant: NeatRacingEvalVariantUnit;
}): NeatRacingEvalWorkerSession {
  const { evaluateVariant } = options;
  // Step 1: Start the phase machine ready; only a round-boundary stop
  // moves it (terminal — the host spins a fresh worker per new round).
  // A state holder (house style, mirroring the sim worker) plus a closure
  // predicate keep TS flow narrowing honest: the phase flips across the
  // closure boundary (handleEval awaits dispatch while handleRequest
  // serves the stop), and CFA cannot see that transition.
  const state: { phase: NeatRacingEvalWorkerPhase } = { phase: 'ready' };
  /** True once a round-boundary stop landed (E11(d) lifecycle flip). */
  const isStopped = (): boolean => state.phase === 'stopped';

  const handleEval = async (
    payload: NeatRacingEvalWorkerEvalPayload,
  ): Promise<NeatRacingEvalWorkerResponse[]> => {
    // Step 2: Contract validation first (single error reply on failure).
    const validation = validateNeatRacingEvalWorkerEvalPayload(payload);
    if (validation !== null) {
      return [createEvalWorkerErrorReply(validation.error)];
    }
    if (isStopped()) {
      return [
        createEvalWorkerErrorReply(
          'eval worker eval failed: session stopped; start a new eval worker for the next adaptation round',
        ),
      ];
    }

    const { seed, requestTick, variants } = payload;
    // Step 3: Bind every slot to its §9.7.2 row-4 child seed up front so
    // the dispatch order never influences the seed derivation order (D2).
    const tasks = variants.map((variant): NeatRacingEvalWorkerTask<number> => {
      const variantSeed = deriveNeatRacingVariantStreamChildSeed(
        seed,
        variant.variantIndex,
        variant.patchSize ?? 0,
      );
      return () => {
        // E11(d): slots opened after a round-boundary stop refuse to
        // run; the whole window's reply is discarded below.
        if (isStopped()) {
          throw new EvalWorkerCancelled();
        }
        return evaluateVariant({ variant, variantSeed, tupleSeed: seed });
      };
    });

    // Step 4: Dispatch under the bounded parallel variant count (TP-18).
    let scores: number[];
    try {
      scores = await runBoundedConcurrency(tasks);
    } catch (thrown: unknown) {
      if (thrown instanceof EvalWorkerCancelled) {
        // Round-boundary cancellation: no reply may escape (E11(d)) — the
        // sim side drops the window and spins a fresh worker per round.
        return [];
      }
      return [
        createEvalWorkerErrorReply(`eval unit failed: ${String(thrown)}`),
      ];
    }

    // Step 5: Belt-and-braces echo guard — a window whose units were all
    // launched pre-stop still posts nothing once the stop landed, so no
    // superseded-round reply can ever reach the sim queue (E11(d)).
    if (isStopped()) {
      return [];
    }

    const resultPayload: NeatRacingEvalWorkerResultPayload = {
      seed,
      requestTick,
      results: variants.map((variant, index) => ({
        variantIndex: variant.variantIndex,
        score: scores[index],
      })),
    };
    return [{ type: 'result', payload: resultPayload }];
  };

  return {
    handleRequest: async (
      request: NeatRacingEvalWorkerRequest,
    ): Promise<NeatRacingEvalWorkerSessionResponse[]> => {
      // Step 1: Malformed wire traffic answers with the single error
      // reply (every message is answered exactly once or cancelled).
      if (typeof request !== 'object' || request === null) {
        return [createEvalWorkerErrorReply('eval worker: malformed message')];
      }
      if (request.type === 'eval') {
        return handleEval(request.payload);
      }
      // Step 2: Round-boundary stop — terminal, reply-free, idempotent.
      // In-flight windows observe the phase flip and discard themselves.
      if (request.type === 'stop') {
        state.phase = 'stopped';
        return [];
      }
      const unknownType = (request as { type?: unknown }).type;
      return [
        createEvalWorkerErrorReply(
          `eval worker: unknown message type ${String(unknownType)}`,
        ),
      ];
    },
    snapshot: (): NeatRacingEvalWorkerSessionSnapshot => ({
      phase: state.phase,
    }),
  };
}

/**
 * Pure-function test seam for the eval worker (P2-S2-A green tests).
 *
 * Educational note: like every established worker module in this repo
 * (flappy evolution, asciiMaze, the neatRacing sim worker), the entrypoint
 * exposes its internals so Jest can drive the protocol contract headless
 * without a worker realm, while production still boots through the module
 * side effect below.
 */
export const NEATRACING_EVAL_WORKER_INTERNALS = {
  createEvalWorkerSession,
  createNeatRacingBootstrapVariantUnit,
} as const;

// Boot: one session per worker realm bound to the bootstrap variant unit
// (the Step 03 NGE growth engine swaps the unit through the session
// option). The async message adapter is the only impurity in this file.
// Responses are posted in order, exactly once each — an eval request posts
// either its result or one error reply (or nothing after a round-boundary
// cancellation), and a stop posts nothing.
const evalWorkerSession = createEvalWorkerSession({
  evaluateVariant: createNeatRacingBootstrapVariantUnit(),
});

self.onmessage = (event: MessageEvent<NeatRacingEvalWorkerRequest>): void => {
  void evalWorkerSession.handleRequest(event.data).then(
    (responses) => {
      for (const response of responses) {
        self.postMessage(response, []);
      }
    },
    (error: unknown) => {
      // Defensive belt: the session resolves every path, so a rejection
      // here can only be a worker bug — keep the wire contract (every
      // message answered exactly once) rather than dropping it silently.
      self.postMessage(
        createEvalWorkerErrorReply(`eval worker failed: ${String(error)}`),
        [],
      );
    },
  );
};
