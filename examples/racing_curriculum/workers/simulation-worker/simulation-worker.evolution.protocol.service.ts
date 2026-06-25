/**
 * Racing worker protocol FSM router.
 *
 * This module owns the `routeRacingWorkerProtocolMessage` function, which maps
 * one host-to-worker message onto the next FSM state and an optional response.
 * It enforces the forward-only lifecycle:
 *
 *   idle → initialised → generation-ready → racing → (stopped)
 *
 * The lifecycle is a straightforward finite-state machine; see
 * [Finite-state machine (Wikipedia)](https://en.wikipedia.org/wiki/Finite-state_machine)
 * for background on why deterministic state transitions help avoid authority
 * confusion between a host and a worker.
 *
 * ## Protocol FSM
 *
 * The diagram below shows the host-to-worker message contract as a state
 * machine.  Each transition is triggered by a single inbound message.  The
 * worker never advances simulation time unless the host asks for a race step,
 * and the host never mutates population state.
 *
 * ```mermaid
 * stateDiagram-v2
 *     [*] --> idle
 *     idle --> initialised : init
 *     initialised --> generation_ready : request-generation
 *     generation_ready --> racing : start-race
 *     racing --> racing : request-race-step
 *     racing --> generation_ready : race finished
 *     idle --> stopped : stop
 *     initialised --> stopped : stop
 *     generation_ready --> stopped : stop
 *     racing --> stopped : stop
 * ```
 *
 * ## Host/worker authority boundary
 *
 * This router runs **inside the worker**.  The host calls `worker.postMessage`
 * with an `RacingWorkerInboundMessage`; the worker calls this router to
 * determine the next FSM state and the outbound payload to send back.
 * The host never advances simulation ticks directly — it only requests steps
 * and renders the snapshots it receives.
 *
 * ## Fallback transport
 *
 * When SharedArrayBuffer or nested worker pools are unavailable (e.g., the
 * host document lacks the required COOP/COEP headers), the same FSM and packed
 * snapshot types work over a single `Worker` with `postMessage`.  No
 * SharedArrayBuffer dependency exists in this module.  The transport layer can
 * be upgraded to a SharedArrayBuffer ring-buffer by wrapping the `postMessage`
 * call site without changing this router.  See
 * [Transferable objects (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Transferring_objects)
 * for the zero-copy transfer semantics used by the generation-ready response.
 *
 * ## Extension points
 *
 * | Hook | Where to extend | Current seam |
 * | --- | --- | --- |
 * | Frozen opponent selection | `simulation-worker.opponent-snapshot.service.ts` | Swap or weight the snapshot sampler |
 * | Generation lifecycle | `simulation-worker.coevolution.service.ts` | Add elitism, diversity pressure, or Lamarckian updates |
 * | Race pack layout | `simulation-worker.race-pack.service.ts` | Add track geometry, curricula, or sensor channels |
 * | Transfer transport | Wrap the `postMessage` caller | Replace structured clone with a ring buffer or batched frames |
 * | Neuromodulation / plasticity | `ModulatorBroadcaster` (not available in Tier 0) | Extend the protocol when dynamic activation or synaptic change primitives are available |
 */
import type {
  EvolutionProtocolRouteResult,
  EvolutionProtocolState,
  GenerationReadyResponse,
  RacingWorkerInboundMessage,
  RacingWorkerPhase,
} from './simulation-worker.evolution.types';

/** Allowed inbound message types per worker phase. */
const PHASE_ALLOWED_MESSAGES: Record<RacingWorkerPhase, readonly string[]> = {
  idle: ['init', 'stop'],
  initialised: ['request-generation', 'stop'],
  'generation-ready': ['start-race', 'stop'],
  racing: ['request-race-step', 'stop'],
  stopped: ['stop'],
};

/**
 * Returns the canonical starting state for the racing worker protocol FSM.
 *
 * The FSM lifecycle is: idle → initialised → generation-ready → racing → stopped.
 *
 * @returns Idle protocol state.
 *
 * @example
 * ```ts
 * const state = createInitialProtocolState();
 * // state.phase === 'idle'
 * ```
 */
export function createInitialProtocolState(): EvolutionProtocolState {
  return { phase: 'idle' };
}

/**
 * Routes one inbound host-to-worker message through the evolution protocol FSM.
 *
 * Messages that arrive in a phase where they are not allowed return an error
 * string and leave `nextState` unchanged.  The `stop` message is always
 * accepted from any phase and unconditionally transitions to `stopped`.
 *
 * Lifecycle transitions:
 * - idle        + init             → initialised
 * - initialised + request-generation → generation-ready
 * - generation-ready + start-race  → racing
 * - racing      + request-race-step → racing (or generation-ready when done)
 * - any phase   + stop             → stopped
 *
 * @param message - Inbound host-to-worker protocol message.
 * @param state - Current FSM state.
 * @returns Next state plus optional response or rejection error.
 */
export function routeRacingWorkerProtocolMessage(
  message: RacingWorkerInboundMessage,
  state: EvolutionProtocolState,
): EvolutionProtocolRouteResult {
  // Step 1: stop is always accepted — transition unconditionally to stopped.
  if (message.type === 'stop') {
    return { nextState: { phase: 'stopped' } };
  }

  // Step 2: Reject messages not permitted in the current phase.
  const allowedTypes = PHASE_ALLOWED_MESSAGES[state.phase];
  if (!allowedTypes.includes(message.type)) {
    return {
      nextState: state,
      error: `Message type '${message.type}' is not allowed in phase '${state.phase}'.`,
    };
  }

  // Step 3: Apply permitted transitions.
  return applyTransition(message, state);

  /** Applies the permitted FSM transition for a validated message. */
  function applyTransition(
    msg: RacingWorkerInboundMessage,
    currentState: EvolutionProtocolState,
  ): EvolutionProtocolRouteResult {
    switch (msg.type) {
      case 'init':
        return { nextState: { phase: 'initialised' } };
      case 'request-generation':
        return {
          nextState: { phase: 'generation-ready' },
          response: createGenerationReadyResponse(),
        };
      case 'start-race':
        return { nextState: { phase: 'racing' } };
      case 'request-race-step':
        return { nextState: currentState };
      default:
        return { nextState: currentState };
    }
  }

  /**
   * Builds a generation-ready response with a placeholder best-network payload.
   *
   * The transfer list includes the ArrayBuffer backing the payload so the host
   * can receive the genome zero-copy over `postMessage`.
   */
  function createGenerationReadyResponse(): GenerationReadyResponse {
    const bestNetworkPayload = new Float32Array([0]);

    return {
      type: 'generation-ready',
      generation: 0,
      teamABestFitness: 0,
      teamBBestFitness: 0,
      bestNetworkPayload,
      transferList: [bestNetworkPayload.buffer],
    };
  }
}
