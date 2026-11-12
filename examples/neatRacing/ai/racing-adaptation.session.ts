/**
 * NeatRacing HOST ADAPTATION SESSION (Phase 4, Step 23, slice
 * `P4-S1-eval-wiring`).
 *
 * THE SESSION (spec section 9.2, AC-043 session half): the AI lane needs ONE
 * owner that spans the whole adaptation lifecycle, and this module is that
 * owner:
 *
 * - it boots ONE eval-worker session per AI-lane lifecycle (the §9.2
 *   two-worker split: the simulation lane stays 60 Hz physics + per-tick
 *   control, everything growth/variant lives in the eval worker);
 * - it drives ONE runtime-adaptation engine per AGENT car (the engines are
 *   created ONCE — their hysteresis, cooldown and AC-030 predicate memory
 *   live in their closures and are never rebuilt per tick);
 * - it evaluates the BOUNDED per-car variant window through the eval worker
 *   (the §9.6 shaped-fitness window: one slot per car, never more than
 *   `NEATRACING_EVAL_WORKER_PARALLEL_VARIANT_COUNT` slots);
 * - it publishes the CURRENTLY adopted per-car network through the
 *   simulation-worker fleet-control seam (the AC-029 adopted-snapshot law:
 *   the relay reads the LIVE network of a car, never a cached copy).
 *
 * LANE ISOLATION (LD-12): car 0 is the HUMAN sample. It owns NO network and
 * its engine is never driven (it is NEVER adapted), but it IS scored in the
 * shaped-fitness window, because the fitness law ranks the WHOLE grid.
 *
 * THE AC-030 OSCILLATION LAW IS NOT RE-DERIVED HERE: the shared predicate
 * lives in `./oscillation-penalty.utils` and the ENGINE evaluates it EXACTLY
 * ONCE per adaptation tick; this session only REPORTS the engagement the
 * engine produced (a second evaluation would advance the hysteresis ramp
 * twice and break the law's single-evaluation invariant).
 *
 * THE EVAL WORKER'S ROLE IN THIS SLICE: the engine's `evaluateScore` hook is
 * SYNCHRONOUS and the engine creates its candidate INTERNALLY, so the async
 * eval worker CANNOT score the engine's candidate before the engine's commit
 * decision. The eval worker therefore evaluates the SESSION-LEVEL variant
 * window (the per-car shaped fitness), while the engine owns growth, its own
 * score windows, exhaustion, commit and rollback.
 *
 * THE ROUND-BOUNDARY LAW: the eval protocol's `stop` request is TERMINAL and
 * discards the in-flight window. A `reset()` therefore stops the outgoing
 * session (so a window that is still in flight can never publish into the
 * new round) and boots a FRESH session — it never leaves a terminal session
 * behind.
 *
 * D1: no `Math.random` anywhere — every stream comes from the AC-026
 * reproducibility tuple (the per-car `car-<i>` network seeds and the
 * `adaptation-<i>` engine seeds).
 *
 * @module
 */

import Network from '../../../src/architecture/network';
import type { TrackSpec } from '../../racing_curriculum/track/track.generator.types';
import {
  NEATRACING_EVAL_WORKER_PARALLEL_VARIANT_COUNT,
  type NeatRacingEvalWorkerResponse,
  type NeatRacingEvalWorkerVariant,
  type NeatRacingEvalWorkerVariantResult,
} from '../workers/eval-worker/eval.protocol';
import {
  createEvalWorkerSession,
  type NeatRacingEvalVariantUnit,
  type NeatRacingEvalWorkerPhase,
  type NeatRacingEvalWorkerSession,
} from '../workers/eval-worker/eval-worker';
import {
  createNeatRacingFleetControl,
  type NeatRacingFleetCarController,
  type NeatRacingFleetControlSeam,
  type NeatRacingFleetControllerRelay,
} from '../workers/simulation-worker/nge.fleet.control';
import {
  deriveNeatRacingCarNetworkChildSeed,
  type NeatRacingReproducibilityTuple,
} from '../workers/simulation-worker/reproducibility.utils';
import { computeNeatRacingShapedFitness } from './fitness.shaping.utils';
import {
  NEATRACING_NGE_CONTROLLER_OUTPUT_COUNT,
  NEATRACING_NGE_DEFAULT_OBSERVATION_LEVEL,
} from './nge.controller';
import {
  NEATRACING_OBSERVATION_TIER_A_CHANNEL_COUNT,
  NEATRACING_OBSERVATION_TIER_B_CHANNEL_COUNT,
  NEATRACING_OBSERVATION_TIER_C_CHANNEL_COUNT,
  type NeatRacingObservationLevel,
} from './observation.assembler';
import type { OscillationEngagement } from './oscillation-penalty.utils';
import {
  createNeatRacingPerCarAdaptationEngines,
  type NeatRacingAdaptationEngine,
  type NeatRacingAdaptationTelemetry,
  type NeatRacingQualitySignal,
} from './runtime.adaptation';

/* ------------------------------------------------------------------------- * *
 * Section 9.2 / AC-043 — the session's named constants
 * ------------------------------------------------------------------------- * */

/**
 * The fleet index of the HUMAN car.
 *
 * Educational note (LD-12 lane isolation): car 0 replays the recorded human
 * sample, so it owns no network and its adaptation engine is never driven.
 * It IS still scored in the shaped-fitness window, because the §9.6 fitness
 * law ranks the whole grid against `maxCars`.
 */
const HUMAN_CAR_INDEX = 0;

/**
 * The observation width of every ladder level (never a magic number).
 *
 * Educational note: the E2 mapping is level -> tier, so level 1 observes
 * Tier A (82 channels), level 2 Tier B (89) and level 3 Tier C (110). The
 * table is the SINGLE mapping of the lane — the session never spells a
 * channel count inline.
 */
const OBSERVATION_WIDTH_BY_LEVEL: Readonly<
  Record<NeatRacingObservationLevel, number>
> = Object.freeze({
  1: NEATRACING_OBSERVATION_TIER_A_CHANNEL_COUNT,
  2: NEATRACING_OBSERVATION_TIER_B_CHANNEL_COUNT,
  3: NEATRACING_OBSERVATION_TIER_C_CHANNEL_COUNT,
});

/**
 * The shared error voice of the adaptation session, matching the lane's
 * `neatRacing <surface> failed` convention.
 */
const NEATRACING_ADAPTATION_SESSION_ERROR_PREFIX =
  'neatRacing adaptation session failed';

/* ------------------------------------------------------------------------- * *
 * Section 9.2 / AC-043 — the session's public types
 * ------------------------------------------------------------------------- * */

/**
 * The construction options of one host adaptation session.
 */
export interface NeatRacingAdaptationSessionOptions {
  /**
   * The AC-026 reproducibility tuple whose seed drives EVERY derived
   * stream of the session (the per-car `car-<i>` network seeds, the
   * `adaptation-<i>` engine seeds and the eval worker's variant seeds).
   */
  readonly tuple: NeatRacingReproducibilityTuple;
  /** The frozen track geometry the fleet-control seam observes. */
  readonly trackSpec: TrackSpec;
  /**
   * The observation ladder level (defaults to the solo rung). The level
   * selects the observation width AND the AC-030 promotion node threshold
   * that gates the oscillation law.
   */
  readonly level?: NeatRacingObservationLevel;
  /**
   * The fleet size (car 0 is the human sample, so the agent cars are
   * `1 .. carCount - 1`). A positive integer.
   */
  readonly carCount: number;
  /**
   * The grid size the shaped fitness ranks the position against (defaults
   * to `carCount`). The §9.6 position term is only invertible relative to
   * the grid it ranks against.
   */
  readonly maxCars?: number;
  /**
   * The fleet slot whose LIVE adopted network backs the HUMAN slot's
   * AI-lane drive. `undefined` (the default) keeps the LD-12 law: the
   * human sample owns no network and the relay publishes nothing for
   * car 0.
   *
   * Educational note — the AI-lane carrier drive: the background NGE
   * carrier mirrors the reference record and is the ONLY session slot
   * whose engine the host names beyond the simulation fleet, so it is
   * the only LIVE "latest adopted" source an AI drive of the human slot
   * can follow (AC-029). Opting in does NOT adapt car 0: its engine
   * stays never-driven and its slot owns no network — the drive BORROWS
   * the carrier's network through the seam (the per-car stream law keeps
   * the borrowed activations deterministic). The simulation lane keeps
   * asking the seam only for the agent cars, so the option only changes
   * consumers that explicitly ask for the human slot.
   *
   * Must name an agent slot strictly after car 0 and within the session
   * fleet (a `RangeError` otherwise).
   */
  readonly humanCarrierDriveSlot?: number;
}

/**
 * The per-car input of one adaptation tick.
 */
export interface NeatRacingAdaptationCarInput {
  /** The fleet index of the car (0 is the human sample). */
  readonly carIndex: number;
  /**
   * The rolling score history the car's own engine consumes (the host
   * assembles it; the engine owns its internal windows).
   */
  readonly scoreHistory: readonly (number | NeatRacingQualitySignal)[];
  /** The car's 1-based finishing position in `[1, maxCars]`. */
  readonly position: number;
  /** The elapsed race time in seconds (`>= 0`). */
  readonly timeSeconds: number;
  /** The clamped lap progress in `[0, 1]`. */
  readonly lapProgress01: number;
}

/**
 * The input of one adaptation tick.
 */
export interface NeatRacingAdaptationSessionTickInput {
  /** The current fixed-timestep tick (a non-negative integer). */
  readonly tick: number;
  /** The per-car inputs of this tick (one entry per fleet car). */
  readonly cars: readonly NeatRacingAdaptationCarInput[];
  /** The current completed lap count (for the `lap_boundary` cadence). */
  readonly completedLaps?: number;
  /** The current completed sector count (for the `sector_boundary` cadence). */
  readonly completedSectors?: number;
}

/**
 * The per-car outcome of one adaptation tick.
 */
export interface NeatRacingAdaptationSessionTickResult {
  /** The fleet index of the car this outcome belongs to. */
  readonly carIndex: number;
  /** Whether the car's engine committed its candidate this tick. */
  readonly committed: boolean;
  /**
   * The deterministic outcome reason (the engine's own vocabulary, plus
   * `human_car_not_adapted` for the human sample).
   */
  readonly reason:
    NeatRacingAdaptationTelemetry['reason'] | 'human_car_not_adapted';
  /**
   * The LIVE node count of the car's network after the tick — the crossing
   * node count the goal-machine feed consumes.
   */
  readonly nodeCount: number;
  /**
   * The shaped fitness the eval worker produced for this car, or null when
   * the bounded window did not cover it.
   */
  readonly shapedFitness: number | null;
  /** The adaptation phase the car's engine reported (growth or stabilization). */
  readonly adaptationPhase: 'growth' | 'stabilization';
  /** The tick index this outcome belongs to. */
  readonly tick: number;
}

/**
 * The plain-data snapshot of one adaptation session.
 */
export interface NeatRacingAdaptationSessionSnapshot {
  /** The tick index of the most recent adaptation tick. */
  readonly tick: number;
  /** The per-car LIVE node counts (indexed by car index). */
  readonly nodeCounts: readonly number[];
  /** The per-car shaped fitness of the most recent window. */
  readonly shapedFitness: readonly (number | null)[];
  /** The eval worker's phase (`ready` until a round-boundary stop lands). */
  readonly evalPhase: NeatRacingEvalWorkerPhase;
  /**
   * The per-car AC-030 oscillation engagement of the most recent tick (the
   * verdict the ENGINE produced — the session never re-evaluates the
   * shared predicate).
   */
  readonly oscillationEngagements: readonly (OscillationEngagement | null)[];
}

/**
 * The host adaptation session surface the AI lane consumes.
 */
export interface NeatRacingAdaptationSession {
  /**
   * The fleet-control seam the simulation lane drives the agent cars
   * through. The seam reads the LIVE adopted network of a car through the
   * session's relay (the AC-029 adopted-snapshot law).
   */
  readonly controller: NeatRacingFleetControlSeam;
  /**
   * Runs ONE adaptation tick: the bounded variant window through the eval
   * worker, then every agent car's own engine.
   *
   * @param input - The tick input (the tick index and the per-car inputs).
   * @returns The per-car outcome of the tick, in input order.
   */
  adaptOnTick(
    input: NeatRacingAdaptationSessionTickInput,
  ): Promise<readonly NeatRacingAdaptationSessionTickResult[]>;
  /**
   * Reads the session's plain-data snapshot.
   *
   * @returns The current tick, the per-car live node counts, the per-car
   *   shaped fitness, the eval-worker phase and the per-car AC-030
   *   engagement.
   */
  snapshot(): NeatRacingAdaptationSessionSnapshot;
  /**
   * Resets the session for a new round: every engine's state, the per-car
   * bookkeeping and the eval-worker session (a FRESH one — the protocol's
   * `stop` request is terminal).
   */
  reset(): void;
}

/* ------------------------------------------------------------------------- * *
 * Section 9.2 / AC-043 — the internal per-car state
 * ------------------------------------------------------------------------- * */

/**
 * The per-car state of one adaptation session.
 */
interface NeatRacingAdaptationCarState {
  /**
   * The car's OWN network, or null for the human sample (car 0 owns no
   * network and is never adapted).
   */
  readonly network: Network | null;
  /** The car's OWN adaptation engine (created once, never rebuilt). */
  readonly engine: NeatRacingAdaptationEngine;
}

/**
 * The driving record one variant slot carries across the eval wire.
 *
 * Educational note: the eval protocol transports the slot payload verbatim
 * (`snapshot: unknown`), so the slot carries the three terms the §9.6
 * shaped fitness is a pure function of.
 */
interface NeatRacingAdaptationDrivingRecord {
  /** The car's 1-based finishing position. */
  readonly position: number;
  /** The elapsed race time in seconds. */
  readonly timeSeconds: number;
  /** The clamped lap progress. */
  readonly lapProgress01: number;
}

/* ------------------------------------------------------------------------- * *
 * Section 9.2 / AC-043 — the session factory (the public surface)
 * ------------------------------------------------------------------------- * */

/**
 * Creates the host adaptation session of one AI-lane lifecycle.
 *
 * Educational note — what the session owns: ONE eval-worker session (the
 * §9.2 growth/variant side), ONE runtime-adaptation engine per agent car
 * (the growth side), the BOUNDED per-car variant window that the eval
 * worker scores with the §9.6 shaped fitness, and the fleet-control seam
 * that publishes the CURRENTLY adopted per-car network to the simulation
 * lane (the AC-029 adopted-snapshot law).
 *
 * LANE ISOLATION (LD-12): car 0 is the human sample — it owns no network
 * and its engine is never driven, but it IS scored in the shaped-fitness
 * window.
 *
 * @param options - The session options (the tuple, the track geometry, the
 *   level, the fleet size and the grid size).
 * @returns The adaptation session (its seam controller, its tick driver, its
 *   snapshot and its reset).
 * @throws {RangeError} With the `neatRacing adaptation session failed` voice
 *   when `carCount` is not a positive integer, when `maxCars` is smaller
 *   than the fleet size or when a tick names a car outside the fleet.
 *
 * @example
 * ```ts
 * const session = createNeatRacingAdaptationSession({
 *   tuple,
 *   trackSpec,
 *   level: 1,
 *   carCount: 3,
 * });
 * const results = await session.adaptOnTick({
 *   tick: 0,
 *   cars: [
 *     { carIndex: 0, scoreHistory: [], position: 2, timeSeconds: 30, lapProgress01: 0.5 },
 *     { carIndex: 1, scoreHistory: [], position: 1, timeSeconds: 28, lapProgress01: 0.6 },
 *   ],
 * });
 * const control = session.controller.computeControl(fleet, 1);
 * ```
 */
export function createNeatRacingAdaptationSession(
  options: NeatRacingAdaptationSessionOptions,
): NeatRacingAdaptationSession {
  // Step 1: Validate the session options (the fleet size and the grid the
  //         shaped fitness ranks against).
  const tuple = options.tuple;
  const trackSpec = options.trackSpec;
  const level = options.level ?? NEATRACING_NGE_DEFAULT_OBSERVATION_LEVEL;
  const carCount = options.carCount;
  const maxCars = options.maxCars ?? carCount;
  const humanCarrierDriveSlot = options.humanCarrierDriveSlot ?? null;
  assertNeatRacingAdaptationSessionOptions(carCount, maxCars);
  // Step 1b: The carrier-drive slot, when opted in, must name an agent slot
  //          strictly after the human slot and within the fleet (the
  //          simulation fleet plus the dedicated background carrier).
  if (
    humanCarrierDriveSlot !== null &&
    (!Number.isInteger(humanCarrierDriveSlot) ||
      humanCarrierDriveSlot <= HUMAN_CAR_INDEX ||
      humanCarrierDriveSlot >= carCount)
  ) {
    throw new RangeError(
      `${NEATRACING_ADAPTATION_SESSION_ERROR_PREFIX}: the human carrier-drive slot must name an agent slot strictly after car 0 and within the fleet of ${String(carCount)} cars, received ${String(humanCarrierDriveSlot)}`,
    );
  }

  // Step 2: Build the per-car state — ONE network and ONE engine per car,
  //         created ONCE (the engines keep their hysteresis, cooldown and
  //         AC-030 predicate memory in their closures).
  const carStates = createNeatRacingAdaptationCarStates(tuple, carCount, level);

  // Step 3: Boot the eval-worker session (ONE per AI-lane lifecycle).
  let evalWorkerSession = createNeatRacingAdaptationEvalWorker(maxCars);

  // Step 4: The per-car live bookkeeping the snapshot reports.
  let currentTick = 0;
  const nodeCountsByCar = new Map<number, number>();
  const shapedFitnessByCar = new Map<number, number>();
  const engagementsByCar = new Map<number, OscillationEngagement>();

  // Step 5: The relay publishes the LIVE adopted network of a car (never a
  //         cached copy — the AC-029 adopted-snapshot law). The human sample
  //         owns no network, so it publishes nothing.
  //
  //         Educational note — the AI-lane human drive (the carrier-backed
  //         slot, the fix packet's ruling (2a)): when the host opts the
  //         session into the background carrier drive, the HUMAN slot's
  //         read is backed by the carrier's OWN live network, so an
  //         AI-lane consumer that asks the seam for the human slot
  //         receives the LATEST adopted carrier snapshot (AC-029) — a
  //         boot network until the carrier's engine commits its first
  //         adoption. The human slot's OWN engine stays never-driven and
  //         it still owns no network (LD-12: the drive BORROWS, it does
  //         not adapt car 0). With no option (the default) the legacy law
  //         is byte-identical: car 0 publishes nothing.
  const relay: NeatRacingFleetControllerRelay = {
    readCarController: (
      carIndex: number,
    ): NeatRacingFleetCarController | null => {
      if (carIndex === HUMAN_CAR_INDEX && humanCarrierDriveSlot !== null) {
        return carStates.get(humanCarrierDriveSlot)?.network ?? null;
      }
      return carStates.get(carIndex)?.network ?? null;
    },
  };

  // Step 6: The production boot injects the seam controller the simulation
  //         lane drives the agent cars through.
  const controller: NeatRacingFleetControlSeam = createNeatRacingFleetControl({
    tupleSeed: tuple.seed,
    trackSpec,
    relay,
    level,
  });

  /**
   * Drives ONE agent car's engine and publishes its live bookkeeping.
   *
   * @param car - The per-car input of this tick.
   * @param tickInput - The tick input (the tick index and the boundaries).
   * @param window - The scored variant window of this tick.
   * @returns The per-car outcome of the tick.
   */
  const adaptCar = async (
    car: NeatRacingAdaptationCarInput,
    tickInput: NeatRacingAdaptationSessionTickInput,
    window: readonly NeatRacingEvalWorkerVariantResult[],
  ): Promise<NeatRacingAdaptationSessionTickResult> => {
    // Step 1: The shaped fitness of this car (null when the bounded window
    //         did not cover it).
    const shapedFitness =
      window.find((entry) => entry.variantIndex === car.carIndex)?.score ??
      null;

    // Step 2: The human sample owns no network and is NEVER adapted.
    const state = carStates.get(car.carIndex);
    const network = state?.network ?? null;
    if (network === null || state === undefined) {
      nodeCountsByCar.set(car.carIndex, 0);
      engagementsByCar.delete(car.carIndex);
      return {
        carIndex: car.carIndex,
        committed: false,
        reason: 'human_car_not_adapted',
        nodeCount: 0,
        shapedFitness,
        adaptationPhase: 'growth',
        tick: tickInput.tick,
      };
    }

    // Step 3: Drive the car's OWN engine with the host's rolling history.
    const telemetry = await state.engine.adaptOnTick({
      tick: tickInput.tick,
      network,
      scoreHistory: car.scoreHistory,
      completedLaps: tickInput.completedLaps,
      completedSectors: tickInput.completedSectors,
    });

    // Step 4: Publish the LIVE node count and the tick's AC-030 engagement
    //         (the engine's own verdict — never re-evaluated here).
    const nodeCount = network.nodes.length;
    nodeCountsByCar.set(car.carIndex, nodeCount);
    engagementsByCar.set(car.carIndex, telemetry.oscillationEngagement);

    return {
      carIndex: car.carIndex,
      committed: telemetry.committed,
      reason: telemetry.reason,
      nodeCount,
      shapedFitness,
      adaptationPhase: telemetry.adaptationPhase,
      tick: tickInput.tick,
    };
  };

  /**
   * Runs ONE adaptation tick: the bounded variant window through the eval
   * worker, then every agent car's own engine.
   *
   * @param tickInput - The tick input (the tick index and the per-car inputs).
   * @returns The per-car outcome of the tick, in input order.
   */
  const adaptOnTick = async (
    tickInput: NeatRacingAdaptationSessionTickInput,
  ): Promise<readonly NeatRacingAdaptationSessionTickResult[]> => {
    // Step 1: Reject a tick that names a car outside the fleet.
    assertNeatRacingAdaptationTickCars(tickInput.cars, carCount);

    // Step 2: Build the bounded variant window (one slot per car).
    const variants = buildNeatRacingAdaptationVariantWindow(tickInput.cars);

    // Step 3: Dispatch the window through the eval worker.
    const responses = await evalWorkerSession.handleRequest({
      type: 'eval',
      payload: { seed: tuple.seed, requestTick: tickInput.tick, variants },
    });

    // Step 4: Read the shaped-fitness scores out of the reply.
    const window = readNeatRacingAdaptationEvalWindow(responses);

    // Step 5: A cancelled round (a reset landed mid-window) publishes
    //         NOTHING — the protocol's own round-boundary law.
    if (window === null) {
      return [];
    }

    // Step 6: Publish the shaped fitness of every scored car.
    for (const entry of window) {
      shapedFitnessByCar.set(entry.variantIndex, entry.score);
    }

    // Step 7: Drive every agent car's engine (car 0 is never adapted).
    const results: NeatRacingAdaptationSessionTickResult[] = [];
    for (const car of tickInput.cars) {
      results.push(await adaptCar(car, tickInput, window));
    }

    // Step 8: Publish the tick index.
    currentTick = tickInput.tick;
    return results;
  };

  /**
   * Reads the session's plain-data snapshot.
   *
   * @returns The current tick, the per-car live node counts, the per-car
   *   shaped fitness, the eval-worker phase and the per-car AC-030
   *   engagement.
   */
  const snapshot = (): NeatRacingAdaptationSessionSnapshot => ({
    tick: currentTick,
    nodeCounts: readNeatRacingAdaptationNumbers(nodeCountsByCar, carCount),
    shapedFitness: readNeatRacingAdaptationNullable(
      shapedFitnessByCar,
      carCount,
    ),
    evalPhase: evalWorkerSession.snapshot().phase,
    oscillationEngagements: readNeatRacingAdaptationNullable(
      engagementsByCar,
      carCount,
    ),
  });

  /**
   * Resets the session for a new round: every engine's state, the per-car
   * bookkeeping and a FRESH eval-worker session.
   *
   * @returns Nothing (the session is reset in place).
   */
  const reset = (): void => {
    // Step 1: Reset every per-car engine (hysteresis, cooldowns, cadence
    //         boundaries and the AC-030 predicate memory).
    for (const state of carStates.values()) {
      state.engine.reset();
    }

    // Step 2: Clear the per-car bookkeeping.
    nodeCountsByCar.clear();
    shapedFitnessByCar.clear();
    engagementsByCar.clear();
    currentTick = 0;

    // Step 3: Stop the outgoing session (the protocol's `stop` is
    //         TERMINAL and discards the in-flight window) and boot a FRESH
    //         one — a reset never leaves a terminal session behind.
    stopNeatRacingAdaptationEvalWorker(evalWorkerSession);
    evalWorkerSession = createNeatRacingAdaptationEvalWorker(maxCars);
  };

  return {
    controller,
    adaptOnTick,
    snapshot,
    reset,
  };
}

/* ------------------------------------------------------------------------- * *
 * Private helpers (below the fold)
 * ------------------------------------------------------------------------- * */

/**
 * Guards the session options: the fleet size must be a positive integer and
 * the grid the shaped fitness ranks against must cover the fleet.
 *
 * @param carCount - The fleet size (car 0 is the human sample).
 * @param maxCars - The grid size the position term ranks against.
 * @throws {RangeError} When either bound is violated.
 */
function assertNeatRacingAdaptationSessionOptions(
  carCount: number,
  maxCars: number,
): void {
  if (!Number.isInteger(carCount) || carCount < 1) {
    throw new RangeError(
      `${NEATRACING_ADAPTATION_SESSION_ERROR_PREFIX}: carCount must be a positive (>= 1) integer, received ${String(carCount)}`,
    );
  }
  if (!Number.isInteger(maxCars) || maxCars < carCount) {
    throw new RangeError(
      `${NEATRACING_ADAPTATION_SESSION_ERROR_PREFIX}: maxCars must be an integer of at least carCount (${carCount}), received ${String(maxCars)}`,
    );
  }
}

/**
 * Guards the per-car inputs of one tick: every car index must name a car of
 * the session's fleet.
 *
 * @param cars - The per-car inputs of the tick.
 * @param carCount - The fleet size of the session.
 * @throws {RangeError} When a car index is outside the fleet.
 */
function assertNeatRacingAdaptationTickCars(
  cars: readonly NeatRacingAdaptationCarInput[],
  carCount: number,
): void {
  for (const car of cars) {
    if (
      !Number.isInteger(car.carIndex) ||
      car.carIndex < 0 ||
      car.carIndex >= carCount
    ) {
      throw new RangeError(
        `${NEATRACING_ADAPTATION_SESSION_ERROR_PREFIX}: car index ${String(car.carIndex)} is outside the session fleet of ${carCount} cars`,
      );
    }
  }
}

/**
 * Builds the per-car state of the session: ONE network and ONE engine per
 * car.
 *
 * Educational note: the human sample (car 0) owns NO network — it replays
 * the recorded human input — while every agent car owns its OWN network
 * instance, seeded from the §9.7.2 `car-<i>` child seed so every car draws
 * from its own stream inside the reproducibility tuple (D1). The engine of
 * car 0 is created (the `adaptation-<i>` table stays aligned with the fleet
 * index) but is NEVER driven.
 *
 * @param tuple - The AC-026 reproducibility tuple.
 * @param carCount - The fleet size.
 * @param level - The observation ladder level (selects the network width).
 * @returns The per-car state map, keyed by car index.
 * @throws {Error} When the per-car engine factory skipped a car.
 */
function createNeatRacingAdaptationCarStates(
  tuple: NeatRacingReproducibilityTuple,
  carCount: number,
  level: NeatRacingObservationLevel,
): Map<number, NeatRacingAdaptationCarState> {
  // Step 1: The engines derive their own `adaptation-<i>` child seeds.
  const engines = createNeatRacingPerCarAdaptationEngines(tuple, carCount, {
    level,
  });

  // Step 2: The observation width of the ladder level (never a magic number).
  const observationWidth = OBSERVATION_WIDTH_BY_LEVEL[level];

  // Step 3: One network per AGENT car (the human sample owns none).
  const states = new Map<number, NeatRacingAdaptationCarState>();
  for (let carIndex = 0; carIndex < carCount; carIndex += 1) {
    const engine = engines.get(carIndex);
    if (engine === undefined) {
      throw new Error(
        `${NEATRACING_ADAPTATION_SESSION_ERROR_PREFIX}: the per-car engine factory skipped car ${carIndex}`,
      );
    }
    states.set(carIndex, {
      network:
        carIndex === HUMAN_CAR_INDEX
          ? null
          : new Network(
              observationWidth,
              NEATRACING_NGE_CONTROLLER_OUTPUT_COUNT,
              {
                seed: deriveNeatRacingCarNetworkChildSeed(tuple.seed, carIndex),
              },
            ),
      engine,
    });
  }
  return states;
}

/**
 * Boots the eval-worker session of the session and binds the §9.6 shaped
 * fitness as its variant evaluation unit.
 *
 * @param maxCars - The grid size the shaped fitness ranks the position
 *   against.
 * @returns The fresh eval-worker session (in the `ready` phase).
 */
function createNeatRacingAdaptationEvalWorker(
  maxCars: number,
): NeatRacingEvalWorkerSession {
  // Step 1: The variant unit scores ONE slot with the shaped fitness
  //         (higher is better) from the driving record the slot carries.
  const evaluateVariant: NeatRacingEvalVariantUnit = ({ variant }) => {
    const record = readNeatRacingAdaptationDrivingRecord(variant);
    return computeNeatRacingShapedFitness(
      record.position,
      record.timeSeconds,
      record.lapProgress01,
      maxCars,
    );
  };

  // Step 2: Boot ONE session per AI-lane lifecycle.
  return createEvalWorkerSession({ evaluateVariant });
}

/**
 * Stops an eval-worker session.
 *
 * Educational note: the protocol's `stop` request is TERMINAL and discards
 * the in-flight window (E11(d)), so a reset that stops the outgoing session
 * guarantees that no superseded round can publish into the new one.
 *
 * @param session - The session to stop.
 * @returns Nothing (the session is stopped in place).
 */
function stopNeatRacingAdaptationEvalWorker(
  session: NeatRacingEvalWorkerSession,
): void {
  void session.handleRequest({ type: 'stop' }).catch(() => undefined);
}

/**
 * Reads the driving record out of one variant slot.
 *
 * Educational note: the eval protocol transports the slot payload verbatim
 * (`snapshot: unknown`), so the unit validates the shape it consumes instead
 * of trusting the wire.
 *
 * @param variant - The variant slot as received on the wire.
 * @returns The driving record of the slot.
 * @throws {Error} When the slot carries no complete driving record.
 */
function readNeatRacingAdaptationDrivingRecord(
  variant: NeatRacingEvalWorkerVariant,
): NeatRacingAdaptationDrivingRecord {
  // Step 1: The opaque payload must be an object.
  const snapshot =
    variant.snapshot as Partial<NeatRacingAdaptationDrivingRecord> | null;
  if (snapshot === null || typeof snapshot !== 'object') {
    throw new Error(
      `${NEATRACING_ADAPTATION_SESSION_ERROR_PREFIX}: variant ${variant.variantIndex} carries no driving record`,
    );
  }

  // Step 2: Every shaped-fitness term must be present.
  const { position, timeSeconds, lapProgress01 } = snapshot;
  if (
    typeof position !== 'number' ||
    typeof timeSeconds !== 'number' ||
    typeof lapProgress01 !== 'number'
  ) {
    throw new Error(
      `${NEATRACING_ADAPTATION_SESSION_ERROR_PREFIX}: variant ${variant.variantIndex} carries an incomplete driving record`,
    );
  }

  return { position, timeSeconds, lapProgress01 };
}

/**
 * Builds the BOUNDED per-car variant window of one tick.
 *
 * Educational note (TP-18): the window holds one slot per car and never
 * more than `NEATRACING_EVAL_WORKER_PARALLEL_VARIANT_COUNT` slots, so the
 * eval worker's dispatch stays inside the adaptation-tick budget.
 *
 * @param cars - The per-car inputs of the tick.
 * @returns The variant slots to dispatch.
 */
function buildNeatRacingAdaptationVariantWindow(
  cars: readonly NeatRacingAdaptationCarInput[],
): readonly NeatRacingEvalWorkerVariant[] {
  // Step 1: Bound the window by the eval worker's parallel variant count.
  const bounded = cars.slice(0, NEATRACING_EVAL_WORKER_PARALLEL_VARIANT_COUNT);

  // Step 2: One slot per car, carrying the driving record the shaped
  //         fitness is computed from.
  return bounded.map((car) => ({
    variantIndex: car.carIndex,
    snapshot: {
      position: car.position,
      timeSeconds: car.timeSeconds,
      lapProgress01: car.lapProgress01,
    },
  }));
}

/**
 * Reads the scored window out of the eval worker's reply.
 *
 * Educational note: the protocol answers a malformed window with ONE error
 * reply (a loud session failure) and answers a cancelled round with NO
 * reply at all (the window published nothing).
 *
 * @param responses - The replies the eval worker returned.
 * @returns The scored slots, or null when the round was cancelled.
 * @throws {Error} When the eval worker rejected the window.
 */
function readNeatRacingAdaptationEvalWindow(
  responses: readonly NeatRacingEvalWorkerResponse[],
): readonly NeatRacingEvalWorkerVariantResult[] | null {
  // Step 1: A protocol-level rejection is a loud session failure.
  const error = responses.find(
    (
      response,
    ): response is Extract<NeatRacingEvalWorkerResponse, { type: 'error' }> =>
      response.type === 'error',
  );
  if (error !== undefined) {
    throw new Error(
      `${NEATRACING_ADAPTATION_SESSION_ERROR_PREFIX}: ${error.payload.message}`,
    );
  }

  // Step 2: A cancelled round answers with no reply at all.
  const result = responses.find(
    (
      response,
    ): response is Extract<NeatRacingEvalWorkerResponse, { type: 'result' }> =>
      response.type === 'result',
  );
  return result?.payload.results ?? null;
}

/**
 * Reads a per-car numeric map as a dense array indexed by car index.
 *
 * @param source - The per-car map.
 * @param carCount - The fleet size.
 * @returns The dense per-car values (0 for a car without an entry).
 */
function readNeatRacingAdaptationNumbers(
  source: ReadonlyMap<number, number>,
  carCount: number,
): readonly number[] {
  return Array.from({ length: carCount }, (_, carIndex) => {
    return source.get(carIndex) ?? 0;
  });
}

/**
 * Reads a per-car map as a dense array indexed by car index, keeping the
 * absent cars explicitly null.
 *
 * @param source - The per-car map.
 * @param carCount - The fleet size.
 * @returns The dense per-car values (null for a car without an entry).
 */
function readNeatRacingAdaptationNullable<T>(
  source: ReadonlyMap<number, T>,
  carCount: number,
): readonly (T | null)[] {
  return Array.from({ length: carCount }, (_, carIndex) => {
    return source.get(carIndex) ?? null;
  });
}
