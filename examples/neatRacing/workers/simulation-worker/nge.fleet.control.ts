/**
 * Per-car NGE fleet-control seam for the neatRacing sim worker (Phase 4,
 * Step 23, slice `P4-S1-worker-seam`).
 *
 * Educational note — what this module owns:
 * The spec §9.2 two-worker split keeps GROWTH (topology mutation, variant
 * evaluation, score windows, commit/rollback bookkeeping) inside the eval
 * worker and keeps INFERENCE inside the sim worker. This module is the sim
 * side of that split: it turns the live fleet state into ONE physics control
 * input per agent car, per tick, through the shipped observe -> activate ->
 * steer chain:
 *
 * 1. OBSERVE — the car's live `PhysicsCarState` is projected onto the
 *    observation seam state (`ai/observation.assembler.ts`), whose level
 *    ladder fixes the vector width (82 / 89 / 110 channels).
 * 2. ACTIVATE — the level-shaped observation vector is pushed through the
 *    activate-only network seam (`ai/nge.controller.ts`), the single owner
 *    of the activation -> control law (output 0 -> throttle, output 1 ->
 *    steer, brake fixed at 0, NO turn-rate channel).
 * 3. STEER — the produced control is published to the sim worker.
 *
 * THE ADOPTED-SNAPSHOT LAW (AC-029): the seam NEVER caches the adopted
 * network. Every tick asks the relay for the CURRENTLY adopted per-car
 * network; when the relay hands back a DIFFERENT network object, the seam
 * seeds it from the §9.7.2 child-seed table (`car-<i>`) and rebuilds its
 * controller over it. The sim lane therefore follows the LATEST adopted
 * snapshot, and a snapshot adopted mid-race reaches the cars on the next
 * tick — a cached snapshot would strand the cars on a stale brain.
 *
 * THE BORROWED-STREAM LAW (the seam does NOT own the adopted network):
 * the adopted network belongs to the EVAL side, so the seam BORROWS its
 * random stream instead of taking ownership of it. On adoption the seam
 * remembers the state the network carried, seeds it from the child-seed
 * table and hands the borrowed state straight back; on every tick the seam
 * restores the CAR's own stream state before the forward pass, re-captures
 * the advanced state afterwards, and returns the network to the state it was
 * found in. Two contractual consequences follow:
 * - every car draws from its OWN stream, even when the relay publishes ONE
 *   network instance for several cars (the per-car stream law), and
 * - the adopted network is left as it was found between ticks, so the sim
 *   lane never silently re-seeds the eval worker's brain.
 * The relay PRECONDITION follows from the same law: the published network
 * instance belongs to ONE car (a per-car-EXCLUSIVE instance). A network
 * without a deterministic stream (a pure activation function) simply skips
 * the stream handling — it has no stream to borrow.
 *
 * THE DETERMINISM RUNG: the seam promises Level 2 (ordered deterministic) —
 * under a FROZEN adopted snapshot, the same tuple seed, the same fleet
 * inputs and the same per-car evaluation order produce byte-stable
 * controls. It does NOT promise Level 3 (replay exact) across an ADOPTION
 * SEQUENCE: the sequence of adopted snapshots is not captured in the AC-026
 * tuple, so a mid-race adoption is a replay boundary.
 *
 * THE ZERO-ALLOCATION LAW (P9): the seam owns NO per-tick garbage. Its
 * per-car working set (the observation state, the roster entry and the
 * published control buffer) is allocated ONCE per car and rewritten in
 * place, so the control object the seam publishes is the SAME instance on
 * every tick. The published buffer is therefore REUSED: the caller copies
 * it before holding it (the sim-worker boundary publishes a fresh physics
 * input per car per tick, so a resolved tick's inputs stay observable after
 * the next tick resolves). The UPSTREAM shipped assembler and controller
 * allocate their own observation vector and control object per call — that
 * cost belongs to those modules, and it is recorded here rather than hidden.
 *
 * @module
 */

import {
  createNeatRacingNgeController,
  NEATRACING_NGE_DEFAULT_OBSERVATION_LEVEL,
  type NeatRacingControllerNetwork,
  type NeatRacingNgeController,
} from '../../ai/nge.controller';
import {
  NEATRACING_TEAM_LAYOUTS,
  type NeatRacingObservationCarState,
  type NeatRacingObservationLevel,
  type NeatRacingObservationState,
} from '../../ai/observation.assembler';
import { NEATRACING_FIXED_TIMESTEP_SECONDS } from '../../host/render-loop.utils';
import type {
  PhysicsCarState,
  PhysicsStepInput,
} from '../../physics/physics.step.service';
import type { TrackSpec } from '../../../racing_curriculum/track/track.generator.types';
import { deriveNeatRacingCarNetworkChildSeed } from './reproducibility.utils';

/**
 * The brake channel the NGE fleet-control seam always publishes: ZERO.
 *
 * Educational note: the shipped NGE controller law (`ai/nge.controller.ts`)
 * maps the network's output channels onto the physics control axes and never
 * commands a brake — the sim lane slows down through the throttle axis only.
 */
export const NEATRACING_FLEET_CONTROL_BRAKE = 0;

/**
 * The per-car network surface the fleet-control seam drives.
 *
 * Educational note: the seam stays inference-only (it never mutates the
 * network's STRUCTURE), but it DOES borrow the network's internal random
 * stream before the first forward pass, so the network's own random draws
 * stay inside the AC-026 reproducibility tuple. `setSeed` is OPTIONAL: a
 * network that carries no internal stream (a pure activation function)
 * simply skips the seeding step. The stream accessors are OPTIONAL too: a
 * network that exposes them lets the seam borrow AND return the stream
 * (the borrowed-stream law), so the adopted network is left exactly as it
 * was found.
 */
export interface NeatRacingFleetCarController extends NeatRacingControllerNetwork {
  /**
   * Seeds the network's internal random stream.
   *
   * @param seed - The unsigned 32-bit child seed for this car's network.
   */
  setSeed?(seed: number): void;
  /**
   * Reads the network's deterministic stream state.
   *
   * @returns The unsigned 32-bit stream state, or undefined when the
   *   network carries no deterministic stream.
   */
  getRNGState?(): number | undefined;
  /**
   * Restores the network's deterministic stream state.
   *
   * @param state - The unsigned 32-bit stream state to restore.
   */
  setRNGState?(state: number): void;
}

/**
 * The relay that publishes the CURRENTLY adopted per-car network.
 *
 * Educational note: the relay is the seam's ONLY window on the eval worker's
 * adopted snapshot. It is read once per car per tick (never cached), so the
 * sim lane follows the latest adoption without any snapshot-copy traffic
 * across the worker boundary.
 *
 * The relay carries a PRECONDITION: the published network instance belongs
 * to ONE car (a per-car-EXCLUSIVE instance). The seam tolerates a shared
 * instance — the per-car stream law keeps every car on its own stream — but
 * the eval side keeps its bookkeeping per car, so an exclusive instance is
 * the intended shape.
 */
export interface NeatRacingFleetControllerRelay {
  /**
   * Reads the currently adopted network of one fleet car.
   *
   * @param carIndex - Fleet car index (0 = the human car).
   * @returns The adopted network, or null when the car has no adopted
   *   network yet (the seam then publishes no control for that car).
   */
  readCarController(carIndex: number): NeatRacingFleetCarController | null;
}

/**
 * The public fleet-control seam the sim worker consumes.
 *
 * Educational note: the published control is a REUSED per-car buffer (the
 * zero-allocation law), so a consumer that HOLDS the control across ticks
 * must copy it first — the sim-worker boundary does exactly that.
 */
export interface NeatRacingFleetControlSeam {
  /**
   * Produces the control input of one fleet car for the next tick.
   *
   * @param fleet - The live fleet states entering this tick.
   * @param carIndex - The fleet car index to control.
   * @returns The published control buffer, or null when the car has no
   *   adopted network yet.
   */
  computeControl(
    fleet: readonly PhysicsCarState[],
    carIndex: number,
  ): PhysicsStepInput | null;
}

/**
 * The construction options of the per-car fleet-control seam.
 */
export interface NeatRacingFleetControlOptions {
  /** AC-026 tuple seed the per-car network child seeds are derived from. */
  readonly tupleSeed: number;
  /** The frozen track geometry the observation assembler reads. */
  readonly trackSpec: TrackSpec;
  /** The relay publishing the currently adopted per-car network. */
  readonly relay: NeatRacingFleetControllerRelay;
  /** Observation ladder level (defaults to the solo NGE level). */
  readonly level?: NeatRacingObservationLevel;
}

/**
 * Maps one produced control onto the physics control input the sim lane
 * publishes.
 *
 * Educational note — the mapping law (AC-029): the two NGE output channels
 * map IDENTITY-wise onto the control axes (channel 0 -> throttle, channel 1
 * -> steer — the shipped `NEATRACING_NGE_CONTROLLER_OUTPUT_COUNT = 2` law),
 * the throttle axis is clamped into `[0, 1]`, the steer axis into
 * `[-1, 1]`, the brake axis is fixed at zero, and NO turn-rate channel is
 * ever emitted (the steer axis is the sole turn channel, AC-093).
 *
 * The mapping returns a FRESH physics input: a resolved tick's inputs must
 * stay observable after the next tick resolves (the adopted-snapshot
 * round-trip law).
 *
 * @param produced - The control the controller produced (null/undefined
 *   when there is nothing to map).
 * @returns The mapped physics control input, or null when there is nothing
 *   to map.
 */
export function mapNeatRacingFleetControl(
  produced: PhysicsStepInput | null | undefined,
): PhysicsStepInput | null {
  if (produced === null || produced === undefined) {
    return null;
  }
  return {
    throttle: clampNeatRacingFleetThrottleAxis(produced.throttle),
    brake: NEATRACING_FLEET_CONTROL_BRAKE,
    steer: clampNeatRacingFleetSteerAxis(produced.steer),
  };
}

/**
 * Creates the per-car NGE fleet-control seam.
 *
 * Educational note: the seam is a small state machine over the adopted
 * snapshot. Its per-car cache holds the adopted network, the controller
 * built over it, the reused observation state and the published control
 * buffer. A tick rebuilds the controller ONLY when the relay hands back a
 * different network object (a fresh adoption), and it reuses everything
 * else — that is where the zero-per-tick-allocation law comes from.
 *
 * @param options - The tuple seed, the frozen track geometry, the relay and
 *   the observation level.
 * @returns The fleet-control seam the sim worker consumes.
 *
 * @example
 * ```ts
 * const seam = createNeatRacingFleetControl({
 *   tupleSeed: 42,
 *   trackSpec,
 *   relay: { readCarController: (carIndex) => adopted.get(carIndex) ?? null },
 * });
 * const control = seam.computeControl(cars, 1);
 * ```
 */
export function createNeatRacingFleetControl(
  options: NeatRacingFleetControlOptions,
): NeatRacingFleetControlSeam {
  const level = options.level ?? NEATRACING_NGE_DEFAULT_OBSERVATION_LEVEL;
  const teamLayout = NEATRACING_TEAM_LAYOUTS[level];
  const carCaches = new Map<number, NeatRacingFleetCarCache>();
  const roster: MutableNeatRacingObservationCarState[] = [];

  return {
    computeControl(
      fleet: readonly PhysicsCarState[],
      carIndex: number,
    ): PhysicsStepInput | null {
      // Step 1: An unknown car slot never publishes a control.
      const focalCar = fleet[carIndex];
      if (!focalCar) {
        return null;
      }

      // Step 2: Ask the relay for the CURRENTLY adopted network (never
      //         cached: the sim lane follows the latest adoption).
      const adopted = options.relay.readCarController(carIndex);
      if (!adopted) {
        return null;
      }

      // Step 3: Adopt the network when it is new to this car (the first
      //         adoption, or a fresh snapshot), re-seeding it from the
      //         §9.7.2 child-seed table.
      const cached = carCaches.get(carIndex);
      const cache =
        cached !== undefined && cached.network === adopted
          ? cached
          : adoptNeatRacingFleetCarController(
              carCaches,
              carIndex,
              adopted,
              options.tupleSeed,
              level,
            );

      // Step 4: Refresh the reused observation working set from the live
      //         fleet (roster first: the rival/teammate tails read it).
      refreshNeatRacingFleetRoster(roster, fleet, teamLayout);
      writeNeatRacingFleetObservationState(
        cache.observation,
        focalCar,
        carIndex,
        resolveNeatRacingFleetTeamIndex(teamLayout, carIndex),
        roster,
      );

      // Step 5: BORROW the car's OWN stream for the forward pass (the
      //         per-car stream law), then RETURN the stream to the state
      //         the adopted network was found in.
      const borrowedState = cache.network.getRNGState?.();
      if (cache.rngState !== undefined) {
        cache.network.setRNGState?.(cache.rngState);
      }

      // Step 6: Observe -> activate -> steer, then publish the mapped
      //         control into the REUSED per-car buffer.
      const produced = cache.controller.computeControl(
        cache.observation,
        options.trackSpec,
      );

      // Step 7: CAPTURE the advanced car stream and RETURN the borrowed
      //         state (the seam never leaves the adopted network
      //         re-seeded).
      cache.rngState = cache.network.getRNGState?.();
      if (borrowedState !== undefined) {
        cache.network.setRNGState?.(borrowedState);
      }

      cache.control.throttle = clampNeatRacingFleetThrottleAxis(
        produced.throttle,
      );
      cache.control.brake = NEATRACING_FLEET_CONTROL_BRAKE;
      cache.control.steer = clampNeatRacingFleetSteerAxis(produced.steer);
      return cache.control;
    },
  };
}

// ── Internal types ──────────────────────────────────────────────────────────

/** Mutable mirror of the observation seam state (the seam rewrites it). */
type MutableNeatRacingObservationState = {
  -readonly [
    K in keyof NeatRacingObservationState
  ]: NeatRacingObservationState[K];
};

/** Mutable mirror of one roster entry (the seam rewrites it). */
type MutableNeatRacingObservationCarState = {
  -readonly [
    K in keyof NeatRacingObservationCarState
  ]: NeatRacingObservationCarState[K];
};

/** The per-car working set of the fleet-control seam. */
interface NeatRacingFleetCarCache {
  /** The currently adopted network of this car. */
  network: NeatRacingFleetCarController;
  /** The controller built over the adopted network. */
  controller: NeatRacingNgeController;
  /** The reused observation state of this car. */
  observation: MutableNeatRacingObservationState;
  /** The reused published control buffer of this car. */
  control: PhysicsStepInput;
  /**
   * The car's OWN stream state (the borrowed sub-stream this car draws
   * from), or undefined when the adopted network carries no deterministic
   * stream.
   */
  rngState: number | undefined;
}

// ── Private helpers ─────────────────────────────────────────────────────────

/**
 * Clamps the throttle axis into its physics domain `[0, 1]`.
 *
 * @param value - The produced throttle value.
 * @returns The clamped throttle (0 for a non-finite value).
 */
function clampNeatRacingFleetThrottleAxis(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  if (value < 0) {
    return 0;
  }
  if (value > 1) {
    return 1;
  }
  return value;
}

/**
 * Clamps the steer axis into its physics domain `[-1, 1]`.
 *
 * @param value - The produced steer value.
 * @returns The clamped steer (0 for a non-finite value).
 */
function clampNeatRacingFleetSteerAxis(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  if (value < -1) {
    return -1;
  }
  if (value > 1) {
    return 1;
  }
  return value;
}

/**
 * Adopts a network for one fleet car: seeds it from the §9.7.2 child-seed
 * table and builds the controller over it.
 *
 * Educational note: the child seed is `mix(tupleSeed, 'car-<i>')`, so every
 * car's network draws from its OWN stream inside the reproducibility tuple
 * (AC-026) — the fleet never shares one stream.
 *
 * The seeding is a BORROW: the adopted network belongs to the eval side, so
 * the seam remembers the state the network carried, seeds it, captures the
 * car's own stream state, and hands the borrowed state straight back.
 *
 * @param carCaches - The seam's per-car cache.
 * @param carIndex - The fleet car index.
 * @param adopted - The newly adopted network.
 * @param tupleSeed - The AC-026 tuple seed.
 * @param level - The observation ladder level.
 * @returns The fresh cache entry for the car (the reused observation state
 *   and control buffer of a previous adoption are carried over).
 */
function adoptNeatRacingFleetCarController(
  carCaches: Map<number, NeatRacingFleetCarCache>,
  carIndex: number,
  adopted: NeatRacingFleetCarController,
  tupleSeed: number,
  level: NeatRacingObservationLevel,
): NeatRacingFleetCarCache {
  // Step 1: BORROW the network's stream: remember the state it carries.
  const borrowedState = adopted.getRNGState?.();

  // Step 2: Seed the network's own stream before its first forward pass.
  adopted.setSeed?.(deriveNeatRacingCarNetworkChildSeed(tupleSeed, carIndex));

  // Step 3: CAPTURE the car's own stream state (its private sub-stream).
  const rngState = adopted.getRNGState?.();

  // Step 4: RETURN the borrowed state (the adopted network is not the
  //         seam's — a donor without a deterministic stream cannot be
  //         returned to its unseeded shape, and keeps the stream the seam
  //         installed).
  if (borrowedState !== undefined) {
    adopted.setRNGState?.(borrowedState);
  }

  // Step 5: Carry the reused working set over (no fresh per-tick garbage).
  const previous = carCaches.get(carIndex);
  const cache: NeatRacingFleetCarCache = {
    network: adopted,
    controller: createNeatRacingNgeController(adopted, { level }),
    observation:
      previous?.observation ?? createNeatRacingFleetObservationState(),
    control: previous?.control ?? {
      throttle: 0,
      brake: NEATRACING_FLEET_CONTROL_BRAKE,
      steer: 0,
    },
    rngState,
  };
  carCaches.set(carIndex, cache);
  return cache;
}

/**
 * Creates the empty (reused) observation state of one fleet car.
 *
 * @returns The fresh observation state.
 */
function createNeatRacingFleetObservationState(): MutableNeatRacingObservationState {
  return {
    carX: 0,
    carY: 0,
    carHeading: 0,
    forwardSpeedWorld: 0,
    lateralSpeedWorld: 0,
    speedWorld: 0,
    yawRateRadiansPerSecond: 0,
    slipAngleRadians: 0,
    teamIndex: 0,
    carIndex: 0,
    cars: undefined,
  };
}

/**
 * Refreshes the reused roster from the live fleet.
 *
 * Educational note: the roster drives the rival and teammate tails of the
 * observation vector; the array index IS the car index, so the focal car is
 * excluded from those tails by its position (the recycled roster contract).
 *
 * @param roster - The reused roster array.
 * @param fleet - The live fleet states entering this tick.
 * @param teamLayout - The level's team layout.
 */
function refreshNeatRacingFleetRoster(
  roster: MutableNeatRacingObservationCarState[],
  fleet: readonly PhysicsCarState[],
  teamLayout: readonly (number | undefined)[],
): void {
  // Step 1: Grow the roster with fresh entries when the fleet grew (the
  //         entries are reused from then on: no per-tick allocation).
  while (roster.length < fleet.length) {
    roster.push(createNeatRacingFleetRosterEntry());
  }
  // Step 2: Shrink the roster when the fleet shrank.
  roster.length = fleet.length;

  // Step 3: Rewrite every entry in place from the live fleet.
  for (let carIndex = 0; carIndex < fleet.length; carIndex += 1) {
    const car = fleet[carIndex];
    const entry = roster[carIndex];
    if (!car || !entry) {
      continue;
    }
    entry.carX = car.x;
    entry.carY = car.y;
    entry.carHeading = car.heading;
    entry.speedWorld = car.speed;
    entry.teamIndex = resolveNeatRacingFleetTeamIndex(teamLayout, carIndex);
  }
}

/**
 * Creates one empty roster entry.
 *
 * @returns The fresh roster entry.
 */
function createNeatRacingFleetRosterEntry(): MutableNeatRacingObservationCarState {
  return {
    carX: 0,
    carY: 0,
    carHeading: 0,
    speedWorld: 0,
    teamIndex: 0,
  };
}

/**
 * Resolves the team index of one fleet car from the level's team layout.
 *
 * @param teamLayout - The level's team layout.
 * @param carIndex - The fleet car index.
 * @returns The team index (0 = team A, 1 = team B).
 */
function resolveNeatRacingFleetTeamIndex(
  teamLayout: readonly (number | undefined)[],
  carIndex: number,
): 0 | 1 {
  const teamIndex = teamLayout[carIndex] ?? 0;
  return teamIndex === 1 ? 1 : 0;
}

/**
 * Writes the live car kinematics into the reused observation state.
 *
 * Educational note — the unit bridge (LD-15): `PhysicsCarState.yawRate` is
 * radians per 60 Hz tick, so the observation seam receives radians per
 * second (`yawRate / NEATRACING_FIXED_TIMESTEP_SECONDS`). The lateral speed
 * is the physics state's own lateral velocity (positive to the left of the
 * heading — the observation seam's convention), so the two sides can never
 * disagree about which way the car slides.
 *
 * @param observation - The reused observation state.
 * @param car - The focal car state.
 * @param carIndex - The focal car's fleet index.
 * @param teamIndex - The focal car's team index.
 * @param roster - The refreshed roster.
 */
function writeNeatRacingFleetObservationState(
  observation: MutableNeatRacingObservationState,
  car: PhysicsCarState,
  carIndex: number,
  teamIndex: 0 | 1,
  roster: readonly NeatRacingObservationCarState[],
): void {
  observation.carX = car.x;
  observation.carY = car.y;
  observation.carHeading = car.heading;
  observation.forwardSpeedWorld = car.speed;
  observation.lateralSpeedWorld = car.lateralVelocity;
  observation.speedWorld = car.speed;
  observation.yawRateRadiansPerSecond =
    car.yawRate / NEATRACING_FIXED_TIMESTEP_SECONDS;
  observation.slipAngleRadians = car.slipAngle;
  observation.teamIndex = teamIndex;
  observation.carIndex = carIndex;
  observation.cars = roster;
}
