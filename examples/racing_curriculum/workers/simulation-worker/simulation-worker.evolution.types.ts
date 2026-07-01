import type { OpponentSnapshotPool } from '../../../../src/neat/nge-collective/neat.nge-collective.types';

/**
 * Typed message union for the worker-authoritative racing evolution protocol.
 *
 * The racing benchmark follows a strict host/worker authority split: the host
 * owns DOM rendering and user interaction; the worker owns environment state,
 * controller inference, Team A/B population containers, and race-step snapshot
 * production.  This file defines the typed envelope for every message that
 * crosses the host↔worker boundary.
 *
 * The two-team, frozen-snapshot evaluation model follows the competitive
 * coevolution pattern.  See [Coevolution (Wikipedia)](https://en.wikipedia.org/wiki/Coevolution)
 * and Stanley & Miikkulainen (2002) on
 * [Neuroevolution of augmenting topologies](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies)
 * for the algorithmic background.
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
 * unchanged.  See [Finite-state machine (Wikipedia)](https://en.wikipedia.org/wiki/Finite-state_machine)
 * for background on the state-machine pattern.
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
 * - Transfer-list resolution for zero-copy `postMessage` frame delivery.  See
 *   [Transferable objects (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Transferring_objects)
 *   for the transfer-list semantics.
 *
 * ## Host-owned responsibilities
 *
 * - DOM layout, canvas rendering, HUD panels, telemetry display.
 * - `requestAnimationFrame` cadence and viewport resize handling.
 * - Receiving and decoding compact `race-step` typed-array snapshots for render.
 * - User interaction (tier selector, keyboard shortcuts).
 *
 * ## Extension points
 *
 * - `TeamPopulationContainer` is a typed handle; the actual NEAT population
 *   wiring will replace the opaque container.
 * - The `generation-ready` worker→host message is typed, but the host consumer
 *   that starts the generation loop is not yet wired end-to-end.
 * - Hall-of-fame + recent opponent sampling is typed, but the network-payload
 *   sampling implementation depends on a real `Neat` snapshot payload.
 * - Radio semantics (`ModulatorBroadcaster`, `EpisodicSlot`, `GatingRouter`)
 *   depend on NGE primitives that are not yet available.
 * - Polyandric reproduction is wired into the generation-boundary transition
 *   in `simulation-worker.evolution.protocol.service.ts` via queen/drone
 *   selection and `reproducePolyandric`. The evolvable-mode flag
 *   (`modeIsEvolvable`) remains descriptor-only until NGE core provides a
 *   runtime operator for it.
 */

/**
 * Lifecycle phases for the racing worker protocol FSM.
 *
 * Phases advance strictly forward: `idle` → `initialised` → `generation-ready`
 * → `racing`.  The `stopped` phase is terminal and reachable from any phase
 * via a `stop` message.
 */
export type RacingWorkerPhase =
  'idle' | 'initialised' | 'generation-ready' | 'racing' | 'stopped';

/**
 * Stateful protocol snapshot carried between inbound worker messages.
 *
 * The FSM router is a pure function: given a message and the current
 * `EvolutionProtocolState`, it returns the next state plus an optional
 * response or rejection error — no shared mutable state required.
 */
export type EvolutionProtocolState = {
  readonly phase: RacingWorkerPhase;
  /** Configuration stored from the init message (no longer silently dropped). */
  readonly initConfig?: {
    readonly populationSize: number;
    readonly rngSeed: number;
    readonly tier: number;
  };
  /** Coevolution container created during request-generation. */
  readonly coevolutionContainer?: unknown;
  /** Per-car adaptation engines created during request-generation. */
  readonly adaptationEngines?: ReadonlyMap<number, unknown>;
  /** Race episode runner created during start-race. */
  readonly raceRunner?: unknown;
  /**
   * Current generation counter.
   *
   * Starts at 0 and increments by 1 each time a race completes and the FSM
   * transitions from `racing` back to `generation-ready`.  Carried in protocol
   * state so the generation-ready response reports the real generation index
   * instead of a hardcoded value.
   */
  readonly generation?: number;
  /**
   * Rolling opponent snapshot store created during the first
   * `request-generation`.  Persists across generations so opponent snapshots
   * accumulate in the hall-of-fame / recent pool rather than being discarded
   * each generation.
   */
  readonly opponentSnapshotStore?: unknown;
  /**
   * Rolling opponent snapshot pool (hall-of-fame) created during the first
   * `request-generation`.  Persists across generations so opponent snapshots
   * accumulate via `addOpponentSnapshot` rather than being discarded each
   * generation.  Uses the core collective `OpponentSnapshotPool` with FIFO
   * eviction.
   */
  readonly opponentSnapshotPool?: OpponentSnapshotPool;
  /**
   * Strategy-divergence tracker that accumulates per-generation team advantage
   * data across the racing coevolution loop. Created during the first
   * `request-generation` and persisted so snapshots accumulate across
   * generations rather than being discarded each generation.
   */
  readonly strategyDivergenceTracker?: StrategyDivergenceTracker;
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
      /** Optional zero-copy transfer list for the generation payload. */
      transferList?: readonly ArrayBuffer[];
      /** Per-car network payloads — one entry per car, not a single shared payload. */
      carNetworkPayloads?: readonly unknown[];
      /** Per-car fitness scores — one entry per car. */
      carFitnessScores?: readonly number[];
      /**
       * Car index whose network is copied back for browser visualization.
       * Must be 0 (blue team #1) per the independent-genome architecture decision.
       */
      visualizationCarIndex?: number;
      /**
       * Serialized network payload for the visualization car (car 0).
       * Must be a real network payload, not a placeholder.
       */
      visualizationPayload?: unknown;
      /**
       * Strategy-divergence classifier result for this generation's team fitness
       * time series. Undefined when fewer than `minGenerations` have been recorded.
       */
      strategyDivergenceClassifier?: StrategyDivergenceClassifierResult;
      /** Per-car pit-lap distribution for Team A (one entry per team member). */
      teamAPitLapDistribution?: PitLapDistribution;
      /** Per-car pit-lap distribution for Team B (one entry per team member). */
      teamBPitLapDistribution?: PitLapDistribution;
    }
  | {
      type: 'race-step';
      requestId: string;
      done: boolean;
      /**
       * Serialized car 0 (blue team #1) network weights for browser visualization.
       * Undefined when no adaptation context is registered.
       */
      visualizationPayload?: Float32Array;
    }
  | { type: 'runtime-status'; phase: RacingWorkerPhase; statusText: string }
  | { type: 'error'; message: string };

/**
 * Typed generation-ready response produced by the worker evolution loop.
 *
 * Carries per-team best fitness and an optional zero-copy payload transfer list.
 */
export type GenerationReadyResponse = {
  readonly type: 'generation-ready';
  readonly generation: number;
  readonly teamABestFitness: number;
  readonly teamBBestFitness: number;
  readonly bestNetworkPayload?: unknown;
  readonly transferList?: readonly ArrayBuffer[];
  /** Per-car network payloads — one entry per car, not a single shared payload. */
  readonly carNetworkPayloads?: readonly unknown[];
  /** Per-car fitness scores — one entry per car. */
  readonly carFitnessScores?: readonly number[];
  /**
   * Car index whose network is copied back for browser visualization.
   * Must be 0 (blue team #1) per the independent-genome architecture decision.
   */
  readonly visualizationCarIndex?: number;
  /**
   * Serialized network payload for the visualization car (car 0).
   * Must be a real network payload, not a placeholder Float32Array([0]).
   */
  readonly visualizationPayload?: unknown;
  /**
   * Strategy-divergence classifier result for this generation's team fitness
   * time series. Undefined when fewer than `minGenerations` have been recorded.
   */
  readonly strategyDivergenceClassifier?: StrategyDivergenceClassifierResult;
  /** Per-car pit-lap distribution for Team A (one entry per team member). */
  readonly teamAPitLapDistribution?: PitLapDistribution;
  /** Per-car pit-lap distribution for Team B (one entry per team member). */
  readonly teamBPitLapDistribution?: PitLapDistribution;
};

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

// ───────────────────────────────────────────────────────────────────────────
// Strategy-divergence analytics types
// ───────────────────────────────────────────────────────────────────────────

/**
 * Per-car pit-lap distribution for one team.
 *
 * Each element is the lap number on which that car entered the pit lane
 * during the race episode. A value of 0 means the car never pitted.
 * The array length equals the team size (e.g. 3 for a 6-car pack).
 */
export type PitLapDistribution = readonly number[];

/**
 * One snapshot of team-level strategy divergence at a generation boundary.
 *
 * Captures the per-team aggregate fitness and per-car pit-lap choices so
 * that the divergence classifier can detect alternating advantage patterns
 * and score how strongly the two teams' strategies are diverging.
 *
 * @property generation — 1-based generation index
 * @property teamAFitness — aggregate fitness for Team A (best or mean)
 * @property teamBFitness — aggregate fitness for Team B (best or mean)
 * @property teamAPitLapDistribution — per-car pit-lap entries for Team A
 * @property teamBPitLapDistribution — per-car pit-lap entries for Team B
 * @property reproductionModeMix — histogram of reproduction modes used
 */
export interface StrategyDivergenceSnapshot {
  readonly generation: number;
  readonly teamAFitness: number;
  readonly teamBFitness: number;
  readonly teamAPitLapDistribution: PitLapDistribution;
  readonly teamBPitLapDistribution: PitLapDistribution;
  /** eslint-disable-next-line @typescript-eslint/no-explicit-any -- histogram bag */
  readonly reproductionModeMix: Record<string, number>;
}

/**
 * Result of classifying a team-fitness time series for strategy divergence.
 *
 * @property isAlternating — true when advantage sign flips on every
 *   consecutive pair (indicates a balanced coevolution arms race)
 * @property dominantPeriod — estimated period of advantage oscillation
 *   (2 when alternating, 1 when one team dominates)
 * @property advantageAmplitude — mean absolute advantage across generations
 * @property divergenceScore — normalised divergence in [0, 1], finite
 */
export interface StrategyDivergenceClassifierResult {
  readonly isAlternating: boolean;
  readonly dominantPeriod: number;
  readonly advantageAmplitude: number;
  readonly divergenceScore: number;
}

/**
 * Accumulator that records per-generation strategy-divergence snapshots
 * and classifies the resulting time series.
 *
 * @property recordSnapshot — append one generation's snapshot
 * @property classify — analyse the accumulated trajectory and return a
 *   classifier result (zeros default when insufficient data)
 * @property getTrajectory — return the read-only snapshot array
 */
export interface StrategyDivergenceTracker {
  readonly recordSnapshot: (snapshot: StrategyDivergenceSnapshot) => void;
  readonly classify: () => StrategyDivergenceClassifierResult;
  readonly getTrajectory: () => readonly StrategyDivergenceSnapshot[];
}

/**
 * Factory service for creating strategy-divergence trackers.
 *
 * @property createStrategyDivergenceTracker — create a new tracker
 *   configured with team size and minimum generation threshold
 */
export interface StrategyDivergenceService {
  readonly createStrategyDivergenceTracker: (config: {
    readonly teamSize: number;
    readonly minGenerations: number;
  }) => StrategyDivergenceTracker;
}
