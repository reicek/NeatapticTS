/**
 * Typed message union for the worker-authoritative racing evolution protocol.
 *
 * The racing benchmark follows a strict host/worker authority split: the host
 * owns DOM rendering and user interaction; the worker owns environment state,
 * controller inference, Team A/B population containers, and race-step snapshot
 * production.  This file defines the typed envelope for every message that
 * crosses the host↔worker boundary.
 *
 * ## Protocol lifecycle
 *
 * The protocol is a forward-only FSM with five phases:
 *
 * ```
 * idle → (init) → initialised → (request-generation) → generation-ready
 *      → (start-race) → racing → (request-race-step*) → racing | generation-ready
 *      any phase → (stop) → stopped
 * ```
 *
 * Messages that arrive in a phase where they are not permitted return an error
 * string from `routeRacingWorkerProtocolMessage` and leave the FSM state
 * unchanged.
 *
 * ## Worker-owned responsibilities
 *
 * - Team A and Team B `Neat` population containers (independent speciation and
 *   fitness tracking).
 * - Rolling opponent snapshot store — frozen at generation start, rotated at
 *   the configured generation boundary.
 * - Generation lifecycle: evaluate against frozen snapshot → select → evolve →
 *   emit `generation-ready` response.
 * - Race episode lifecycle: build a deterministic race pack → tick simulation →
 *   run controller inference per car per tick → stream packed `race-step`
 *   typed-array snapshots.
 * - Transfer-list resolution for zero-copy `postMessage` frame delivery.
 *
 * ## Host-owned responsibilities
 *
 * - DOM layout, canvas rendering, HUD panels, telemetry display.
 * - `requestAnimationFrame` cadence and viewport resize handling.
 * - Receiving and decoding compact `race-step` typed-array snapshots for render.
 * - User interaction (tier selector, keyboard shortcuts).
 *
 * ## Remaining gaps
 *
 * - Real `Neat` population wiring (speciation, fitness, genome selection) is
 *   deferred; `TeamPopulationContainer` is currently an opaque handle.
 * - `generation-ready` worker→host message is typed but the generation loop is
 *   not yet wired into the host.
 * - Rolling opponent snapshot hall-of-fame + recent-sample selection is a
 *   placeholder; no actual network payload sampling exists yet.
 * - Radio semantics (`ModulatorBroadcaster`, `EpisodicSlot`, `GatingRouter`)
 *   depend on upstream NGE Phase G primitives — marked as `NGE_TODO` in the
 *   coevolution and race-pack service files.
 * - Polyandric reproduction (`modeIsEvolvable`, upstream NGE Phase E) is not
 *   yet available.
 */

/**
 * Lifecycle phases for the racing worker protocol FSM.
 *
 * Phases advance strictly forward: `idle` → `initialised` → `generation-ready`
 * → `racing`.  The `stopped` phase is terminal and reachable from any phase
 * via a `stop` message.
 */
export type RacingWorkerPhase =
  | 'idle'
  | 'initialised'
  | 'generation-ready'
  | 'racing'
  | 'stopped';

/**
 * Stateful protocol snapshot carried between inbound worker messages.
 *
 * The FSM router is a pure function: given a message and the current
 * `EvolutionProtocolState`, it returns the next state plus an optional
 * response or rejection error — no shared mutable state required.
 */
export type EvolutionProtocolState = {
  readonly phase: RacingWorkerPhase;
};

/**
 * Host-to-worker messages for the racing evolution protocol.
 *
 * Messages are accepted only in the phase where they are permitted.
 * Sending `request-generation` before `init`, or `start-race` before a
 * generation has completed, produces a rejection error instead of a
 * transition.  The `stop` message is always accepted and transitions to
 * `stopped` unconditionally.
 */
export type RacingWorkerInboundMessage =
  | { type: 'init'; populationSize: number; rngSeed: number; tier: number }
  | { type: 'request-generation' }
  | { type: 'start-race'; tierConfig: unknown; opponentSnapshotId: string }
  | { type: 'request-race-step'; requestId: string; stepsToAdvance: number }
  | { type: 'stop' };

/**
 * Worker-to-host outbound messages for the racing evolution protocol.
 *
 * These represent the compact, streaming payloads the worker sends back to the
 * host after each protocol event:
 * - `generation-ready` — emitted after a full evolution generation completes;
 *   carries per-team best fitness and optionally the best genome payload.
 * - `race-step` — compact typed-array snapshot for one or more simulation
 *   ticks; the host should render each snapshot at display cadence.
 * - `runtime-status` — human-readable status text for HUD display.
 * - `error` — protocol or runtime error; the host should surface this to the
 *   user and consider stopping.
 *
 * Note: `race-step` snapshots must be transferred with a transfer list so that
 * typed-array buffers are zero-copy handed to the host thread.  See
 * `resolveRaceStepTransferList` in the race-pack service.
 */
export type RacingWorkerOutboundMessage =
  | {
      type: 'generation-ready';
      generation: number;
      teamABestFitness: number;
      teamBBestFitness: number;
      bestNetworkPayload?: unknown;
    }
  | {
      type: 'race-step';
      requestId: string;
      done: boolean;
    }
  | { type: 'runtime-status'; phase: RacingWorkerPhase; statusText: string }
  | { type: 'error'; message: string };

/**
 * Route result returned by `routeRacingWorkerProtocolMessage` for one inbound
 * message.
 *
 * - `nextState` — updated FSM state to use for the next message.
 * - `response` — optional outbound payload the worker should send to the host.
 * - `error` — rejection reason when the message was not permitted in the
 *   current phase; `nextState` is unchanged when `error` is set.
 */
export type EvolutionProtocolRouteResult = {
  readonly nextState: EvolutionProtocolState;
  readonly response?: unknown;
  readonly error?: string;
};
