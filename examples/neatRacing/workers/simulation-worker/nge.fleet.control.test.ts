/**
 * Owner-local test surface for the NGE fleet-control seam (Phase 4, Step 23,
 * slice `P4-S1-worker-seam`).
 *
 * Educational note — what this file owns:
 * The RED home (`simulation-worker.control-seam.test.ts`) pins the OBSERVABLE
 * control stream of the sim lane: the agent cars follow the INJECTED
 * controller, car 0 keeps the replayed human sample, and the lane follows the
 * LATEST adopted snapshot. It deliberately does NOT pin HOW a network becomes
 * a control, so the production accessor names stay free.
 *
 * This file owns the OTHER half — the seam's OWN laws:
 * - the §9.7.2 per-car network child-seed table (`car-<i>` per car index),
 * - the activation -> control mapping law (output channel 0 -> throttle,
 *   output channel 1 -> steer, throttle clamped `[0, 1]`, steer clamped
 *   `[-1, 1]`, brake fixed at 0 and NO turn-rate channel),
 * - the observe -> activate chain (ONE level-shaped forward pass per tick),
 * - the zero-per-frame-allocation guard (ONE reused control buffer per car),
 * - the borrowed-stream law (every car draws from its OWN stream even when
 *   the relay shares ONE network instance, and the adopted network is left
 *   exactly as the seam found it),
 * - the frozen-snapshot byte-parity over `NEATRACING_REPLAY_ADAPTATION_TICKS`
 *   ticks.
 *
 * Every fixture is DETERMINISTIC: fixed seeds, the SHIPPED autopilot fixture
 * track, PURE activation functions, NO RNG, NO timers, NO wall-clock reads.
 *
 * @module
 */

import { describe, expect, it } from '@jest/globals';

import { resolveNeatRacingObservationChannelCount } from '../../ai/observation.assembler';
import { createTrackSpecForAutopilotFixture } from '../../physics/ai.autopilot.fixtures';
import { createInitialPhysicsCarState } from '../../physics/physics.step.service';
import type {
  PhysicsCarState,
  PhysicsStepInput,
} from '../../physics/physics.step.service';
import type { TrackSpec } from '../../../racing_curriculum/track/track.generator.types';
import {
  createNeatRacingFleetControl,
  mapNeatRacingFleetControl,
  NEATRACING_FLEET_CONTROL_BRAKE,
  type NeatRacingFleetCarController,
  type NeatRacingFleetControlSeam,
  type NeatRacingFleetControllerRelay,
} from './nge.fleet.control';
import {
  deriveNeatRacingCarNetworkChildSeed,
  NEATRACING_REPLAY_ADAPTATION_TICKS,
} from './reproducibility.utils';

/** The AC-026 tuple seed the fixture fleet derives its per-car child seeds from. */
const FLEET_CONTROL_TEST_TUPLE_SEED = 42;

/** The observation ladder level the fixture seam observes at (the solo rung). */
const FLEET_CONTROL_TEST_LEVEL = 1;

/** The distance one fixture tick advances a car per unit of throttle. */
const FLEET_CONTROL_TEST_ADVANCE_M_PER_THROTTLE = 1;

/** The forward speed one fixture tick assigns per unit of throttle. */
const FLEET_CONTROL_TEST_SPEED_PER_THROTTLE = 12;

/**
 * Builds the fixture track: the SHIPPED autopilot fixture line (one straight
 * closed loop of 40 lane-center samples along +x at y = 0) plus the segment
 * the observation assembler and the spawn resolver read.
 *
 * @returns The frozen fixture track geometry.
 */
function createFleetControlTestTrackSpec(): TrackSpec {
  return {
    seed: FLEET_CONTROL_TEST_TUPLE_SEED,
    layoutVersion: 1,
    sizeBucket: 'fleet-control-fixture',
    segments: [{ startX: 0, startY: 0, endX: 39, endY: 0, width: 12 }],
    splineSamples: createTrackSpecForAutopilotFixture(),
    laneCount: 2,
  };
}

/**
 * Builds the fixture fleet: car 0 is the HUMAN car (no adopted network), cars
 * 1 and 2 are the AGENT cars the seam drives.
 *
 * @returns The three deterministic fixture car states.
 */
function createFleetControlTestFleet(): PhysicsCarState[] {
  return [
    createInitialPhysicsCarState(42, { x: 2, y: 0, heading: 0 }),
    createInitialPhysicsCarState(43, { x: 14, y: 0, heading: 0 }),
    createInitialPhysicsCarState(44, { x: 26, y: 0, heading: 0 }),
  ];
}

/** The fixture network: a PURE activation function plus the seam's call log. */
type FleetControlTestNetwork = NeatRacingFleetCarController & {
  /** The child seeds the seam pushed into this network (in call order). */
  readonly seeds: number[];
  /** The observation vectors this network saw (in call order). */
  readonly vectors: readonly (readonly number[])[];
};

/**
 * Creates a fixture network whose forward pass is a PURE function of the
 * observation vector (no RNG, no clock, no internal state).
 *
 * @param activations - The pure activation law of the network.
 * @returns The fixture network.
 */
function createFleetControlTestNetwork(
  activations: (vector: readonly number[]) => readonly number[],
): FleetControlTestNetwork {
  const seeds: number[] = [];
  const vectors: (readonly number[])[] = [];
  return {
    seeds,
    vectors,
    setSeed: (seed: number): void => {
      seeds.push(seed);
    },
    activate: (vector: readonly number[] | Float32Array): readonly number[] => {
      const snapshot = Array.from(vector);
      vectors.push(snapshot);
      return activations(snapshot);
    },
  };
}

/**
 * The fixture network that models the SHIPPED deterministic network stream.
 *
 * Educational note: the fixture keeps ONE unsigned 32-bit stream state, so
 * `setSeed` installs a stream, `getRNGState`/`setRNGState` read and restore
 * it, and every forward pass draws ONE value from it. The draw law is a
 * plain LCG, so the expected draw sequence of a child seed is reproducible
 * from the seed alone — the seam's own stream handling is falsifiable.
 */
type FleetControlTestStreamNetwork = NeatRacingFleetCarController & {
  /** The draws the network produced (in call order). */
  readonly draws: number[];
  /** The child seeds the seam pushed into this network (in call order). */
  readonly seeds: number[];
};

/**
 * Advances the fixture stream by one LCG step.
 *
 * @param state - The stream state before the draw.
 * @returns The stream state after the draw.
 */
function advanceFleetControlTestStream(state: number): number {
  return (Math.imul(state, 1664525) + 1013904223) >>> 0;
}

/**
 * Normalizes a fixture stream state into its drawn value.
 *
 * @param state - The stream state after a draw.
 * @returns The drawn value in `[0, 1)`.
 */
function fleetControlTestStreamDraw(state: number): number {
  return state / 4294967296;
}

/**
 * Creates the fixture network that carries a deterministic stream.
 *
 * @returns The fixture stream network.
 */
function createFleetControlTestStreamNetwork(): FleetControlTestStreamNetwork {
  const seeds: number[] = [];
  const draws: number[] = [];
  let state: number | undefined;
  return {
    seeds,
    draws,
    setSeed: (seed: number): void => {
      seeds.push(seed);
      state = seed >>> 0;
    },
    getRNGState: (): number | undefined => state,
    setRNGState: (next: number): void => {
      state = next >>> 0;
    },
    activate: (): readonly number[] => {
      state = advanceFleetControlTestStream(state ?? 0);
      const draw = fleetControlTestStreamDraw(state);
      draws.push(draw);
      return [draw, draw];
    },
  };
}

/**
 * Reproduces the draw sequence a fixture stream yields from one seed.
 *
 * @param seed - The stream seed (a §9.7.2 child seed).
 * @param count - The number of draws to reproduce.
 * @returns The reproduced draw sequence.
 */
function createFleetControlTestReferenceStream(
  seed: number,
  count: number,
): readonly number[] {
  let state = seed >>> 0;
  const sequence: number[] = [];
  for (let index = 0; index < count; index += 1) {
    state = advanceFleetControlTestStream(state);
    sequence.push(fleetControlTestStreamDraw(state));
  }
  return sequence;
}

/** The fixture relay plus the publisher that swaps the adopted snapshot. */
type FleetControlTestRelay = {
  /** The relay the seam reads the currently adopted network from. */
  readonly relay: NeatRacingFleetControllerRelay;
  /**
   * Publishes (or clears) the adopted network of one car.
   *
   * @param carIndex - The fleet car index.
   * @param network - The newly adopted network, or null to clear it.
   */
  publish(carIndex: number, network: NeatRacingFleetCarController | null): void;
};

/**
 * Creates the fixture relay: a tiny in-memory adopted-snapshot store.
 *
 * @returns The fixture relay and its publisher.
 */
function createFleetControlTestRelay(): FleetControlTestRelay {
  const adopted = new Map<number, NeatRacingFleetCarController>();
  return {
    relay: {
      readCarController: (
        carIndex: number,
      ): NeatRacingFleetCarController | null => adopted.get(carIndex) ?? null,
    },
    publish: (
      carIndex: number,
      network: NeatRacingFleetCarController | null,
    ): void => {
      if (network === null) {
        adopted.delete(carIndex);
        return;
      }
      adopted.set(carIndex, network);
    },
  };
}

/**
 * Creates the fixture seam over the given relay.
 *
 * @param relay - The relay publishing the adopted per-car networks.
 * @returns The seam under test.
 */
function createFleetControlTestSeam(
  relay: NeatRacingFleetControllerRelay,
): NeatRacingFleetControlSeam {
  return createNeatRacingFleetControl({
    tupleSeed: FLEET_CONTROL_TEST_TUPLE_SEED,
    trackSpec: createFleetControlTestTrackSpec(),
    relay,
    level: FLEET_CONTROL_TEST_LEVEL,
  });
}

/**
 * The fixture activation law: a PURE function of the observation vector.
 *
 * Educational note: the law reads two channels of the assembled vector, so
 * the published control CHANGES as the fleet moves — the byte-parity contract
 * below therefore compares a LIVE control stream, not a constant.
 *
 * @param vector - The assembled observation vector.
 * @returns The two output channels (throttle, steer).
 */
function createFleetControlTestActivations(
  vector: readonly number[],
): readonly number[] {
  return [
    0.5 + 0.25 * Math.sin(vector[0] ?? 0),
    0.5 * Math.cos(vector[1] ?? 0),
  ];
}

/**
 * Advances the fixture fleet by one tick of the published controls.
 *
 * @param fleet - The live fixture fleet (mutated in place).
 * @param controls - The published controls of this tick, per car.
 */
function advanceFleetControlTestFleet(
  fleet: PhysicsCarState[],
  controls: readonly (PhysicsStepInput | null)[],
): void {
  for (let carIndex = 0; carIndex < fleet.length; carIndex += 1) {
    const control = controls[carIndex];
    if (control === null) {
      continue;
    }
    const car = fleet[carIndex];
    car.x += control.throttle * FLEET_CONTROL_TEST_ADVANCE_M_PER_THROTTLE;
    car.speed = control.throttle * FLEET_CONTROL_TEST_SPEED_PER_THROTTLE;
  }
}

/**
 * Runs ONE frozen-snapshot fleet session and collects the published control
 * stream of every car.
 *
 * Educational note: the published control buffer is REUSED across ticks (the
 * zero-allocation law), so the stream COPIES each control before holding it —
 * exactly what the sim-worker boundary does at its own seam.
 *
 * @returns The per-tick control stream of the whole fleet.
 */
function collectFrozenSnapshotControlStream(): readonly (readonly (PhysicsStepInput | null)[])[] {
  const relay = createFleetControlTestRelay();
  relay.publish(
    1,
    createFleetControlTestNetwork(createFleetControlTestActivations),
  );
  relay.publish(
    2,
    createFleetControlTestNetwork(createFleetControlTestActivations),
  );
  const seam = createFleetControlTestSeam(relay.relay);
  const fleet = createFleetControlTestFleet();
  const stream: (PhysicsStepInput | null)[][] = [];

  for (let tick = 0; tick < NEATRACING_REPLAY_ADAPTATION_TICKS; tick += 1) {
    const controls = fleet.map((_, carIndex) =>
      seam.computeControl(fleet, carIndex),
    );
    stream.push(
      controls.map((control) => (control === null ? null : { ...control })),
    );
    advanceFleetControlTestFleet(fleet, controls);
  }

  return stream;
}

describe('the NGE fleet-control seam (P4-S1-worker-seam)', () => {
  describe('the §9.7.2 per-car network child-seed table', () => {
    it('seeds every car network from its OWN car-<i> child seed', () => {
      // Arrange: one adopted network per agent car.
      const relay = createFleetControlTestRelay();
      const carOneNetwork = createFleetControlTestNetwork(() => [0.4, 0.1]);
      const carTwoNetwork = createFleetControlTestNetwork(() => [0.4, 0.1]);
      relay.publish(1, carOneNetwork);
      relay.publish(2, carTwoNetwork);
      const seam = createFleetControlTestSeam(relay.relay);
      const fleet = createFleetControlTestFleet();

      // Act: resolve one tick for both agent cars.
      seam.computeControl(fleet, 1);
      seam.computeControl(fleet, 2);

      // Assert: each car network carries its OWN child seed.
      expect(carOneNetwork.seeds).toEqual([
        deriveNeatRacingCarNetworkChildSeed(FLEET_CONTROL_TEST_TUPLE_SEED, 1),
      ]);
      expect(carTwoNetwork.seeds).toEqual([
        deriveNeatRacingCarNetworkChildSeed(FLEET_CONTROL_TEST_TUPLE_SEED, 2),
      ]);
    });

    it('re-seeds the car network when a NEW snapshot is adopted', () => {
      // Arrange: one adopted network, then a second one.
      const relay = createFleetControlTestRelay();
      const retiredNetwork = createFleetControlTestNetwork(() => [0.31, -0.73]);
      const adoptedNetwork = createFleetControlTestNetwork(() => [0.62, 0.44]);
      relay.publish(1, retiredNetwork);
      const seam = createFleetControlTestSeam(relay.relay);
      const fleet = createFleetControlTestFleet();

      // Act: resolve one tick, adopt the second snapshot, resolve again.
      seam.computeControl(fleet, 1);
      relay.publish(1, adoptedNetwork);
      seam.computeControl(fleet, 1);

      // Assert: the NEW snapshot is seeded from the same car-<i> child seed
      // and the retired snapshot is never re-seeded.
      expect(adoptedNetwork.seeds).toEqual([
        deriveNeatRacingCarNetworkChildSeed(FLEET_CONTROL_TEST_TUPLE_SEED, 1),
      ]);
      expect(retiredNetwork.seeds).toHaveLength(1);
    });
  });

  describe('the activation -> control mapping law', () => {
    it('maps the two output channels IDENTITY-wise onto throttle and steer', () => {
      // Arrange: a network whose output channels are (0.42, -0.37).
      const relay = createFleetControlTestRelay();
      relay.publish(
        1,
        createFleetControlTestNetwork(() => [0.42, -0.37]),
      );
      const seam = createFleetControlTestSeam(relay.relay);
      const fleet = createFleetControlTestFleet();

      // Act: resolve one tick for the agent car.
      const control = seam.computeControl(fleet, 1);

      // Assert: channel 0 became the throttle and channel 1 the steer.
      expect(control).toEqual({ throttle: 0.42, brake: 0, steer: -0.37 });
    });

    it('clamps the throttle axis into [0,1] and the steer axis into [-1,1]', () => {
      // Arrange: a network whose output channels are out of domain.
      const relay = createFleetControlTestRelay();
      relay.publish(
        1,
        createFleetControlTestNetwork(() => [1.8, -2.4]),
      );
      const seam = createFleetControlTestSeam(relay.relay);
      const fleet = createFleetControlTestFleet();

      // Act: resolve one tick for the agent car.
      const control = seam.computeControl(fleet, 1);

      // Assert: both axes landed inside their physics domain.
      expect(control).toEqual({ throttle: 1, brake: 0, steer: -1 });
    });

    it('never emits a turn-rate channel and always reports brake 0', () => {
      // Arrange: an ordinary network.
      const relay = createFleetControlTestRelay();
      relay.publish(
        1,
        createFleetControlTestNetwork(() => [0.5, 0.25]),
      );
      const seam = createFleetControlTestSeam(relay.relay);
      const fleet = createFleetControlTestFleet();

      // Act: resolve one tick for the agent car.
      const control = seam.computeControl(fleet, 1);

      // Assert: the steer axis is the sole turn channel (AC-093) and the
      // brake axis stays at zero.
      expect(control?.turnRatePercent).toBeUndefined();
      expect(control?.brake).toBe(NEATRACING_FLEET_CONTROL_BRAKE);
    });

    it('maps a produced control onto the clamped control axes', () => {
      // Act: map an out-of-domain produced control.
      const mapped = mapNeatRacingFleetControl({
        throttle: 2.5,
        brake: 0.9,
        steer: -3,
      });

      // Assert: the mapped control carries the clamped axes, brake 0 and no
      // turn-rate channel.
      expect(mapped).toEqual({ throttle: 1, brake: 0, steer: -1 });
    });
  });

  describe('the observe -> activate -> steer chain', () => {
    it('feeds ONE level-shaped observation vector into the network per tick', () => {
      // Arrange: one adopted network.
      const relay = createFleetControlTestRelay();
      const network = createFleetControlTestNetwork(() => [0.5, 0]);
      relay.publish(1, network);
      const seam = createFleetControlTestSeam(relay.relay);
      const fleet = createFleetControlTestFleet();

      // Act: resolve one tick for the agent car.
      seam.computeControl(fleet, 1);

      // Assert: exactly ONE forward pass, over the level-shaped vector.
      expect(network.vectors).toHaveLength(1);
      expect(network.vectors[0]).toHaveLength(
        resolveNeatRacingObservationChannelCount(FLEET_CONTROL_TEST_LEVEL),
      );
    });

    it('follows the LATEST adopted snapshot', () => {
      // Arrange: one adopted network, then a second one.
      const relay = createFleetControlTestRelay();
      relay.publish(
        1,
        createFleetControlTestNetwork(() => [0.31, -0.73]),
      );
      const seam = createFleetControlTestSeam(relay.relay);
      const fleet = createFleetControlTestFleet();

      // Act + assert: the first tick follows the first snapshot...
      expect(seam.computeControl(fleet, 1)).toEqual({
        throttle: 0.31,
        brake: 0,
        steer: -0.73,
      });

      // Act + assert: ...and the next tick follows the SECOND one.
      relay.publish(
        1,
        createFleetControlTestNetwork(() => [0.62, 0.44]),
      );
      expect(seam.computeControl(fleet, 1)).toEqual({
        throttle: 0.62,
        brake: 0,
        steer: 0.44,
      });
    });

    it('publishes no control before a network is adopted', () => {
      // Arrange: a relay with no adopted network.
      const relay = createFleetControlTestRelay();
      const seam = createFleetControlTestSeam(relay.relay);
      const fleet = createFleetControlTestFleet();

      // Act: resolve one tick for the agent car.
      const control = seam.computeControl(fleet, 1);

      // Assert: the seam publishes nothing (the caller owns the fallback).
      expect(control).toBeNull();
    });
  });

  describe('the zero-per-frame-allocation guard', () => {
    it('reuses ONE published control buffer per car across ticks', () => {
      // Arrange: one adopted network.
      const relay = createFleetControlTestRelay();
      relay.publish(
        1,
        createFleetControlTestNetwork(() => [0.5, 0]),
      );
      const seam = createFleetControlTestSeam(relay.relay);
      const fleet = createFleetControlTestFleet();

      // Act: resolve two ticks for the same agent car.
      const first = seam.computeControl(fleet, 1);
      const second = seam.computeControl(fleet, 1);

      // Assert: the published buffer is the SAME instance (no per-tick
      // allocation inside the seam).
      expect(second).toBe(first);
    });

    it('runs exactly ONE forward pass per car per tick', () => {
      // Arrange: one adopted network.
      const relay = createFleetControlTestRelay();
      const network = createFleetControlTestNetwork(() => [0.5, 0]);
      relay.publish(1, network);
      const seam = createFleetControlTestSeam(relay.relay);
      const fleet = createFleetControlTestFleet();

      // Act: resolve three ticks for the same agent car.
      seam.computeControl(fleet, 1);
      seam.computeControl(fleet, 1);
      seam.computeControl(fleet, 1);

      // Assert: one forward pass per tick, no rebuild churn.
      expect(network.vectors).toHaveLength(3);
    });
  });

  describe('the frozen-snapshot byte-parity', () => {
    it('produces byte-identical control streams over the replay window', () => {
      // Act: run the SAME frozen-snapshot session twice.
      const streamA = collectFrozenSnapshotControlStream();
      const streamB = collectFrozenSnapshotControlStream();

      // Assert: the two control streams are byte-identical.
      expect(JSON.stringify(streamA)).toBe(JSON.stringify(streamB));
    });
  });

  describe('the borrowed-stream law', () => {
    it('keeps every car on its OWN stream when the relay shares ONE instance', () => {
      // Arrange: ONE network instance published for BOTH agent cars.
      const relay = createFleetControlTestRelay();
      const sharedNetwork = createFleetControlTestStreamNetwork();
      relay.publish(1, sharedNetwork);
      relay.publish(2, sharedNetwork);
      const seam = createFleetControlTestSeam(relay.relay);
      const fleet = createFleetControlTestFleet();

      // Act: resolve TWO ticks for both agent cars.
      seam.computeControl(fleet, 1);
      seam.computeControl(fleet, 2);
      seam.computeControl(fleet, 1);
      seam.computeControl(fleet, 2);

      // Assert: car 1 drew from the `car-1` child-seed stream and car 2 from
      // the `car-2` one — a shared instance never couples the two cars.
      const carOneStream = createFleetControlTestReferenceStream(
        deriveNeatRacingCarNetworkChildSeed(FLEET_CONTROL_TEST_TUPLE_SEED, 1),
        2,
      );
      const carTwoStream = createFleetControlTestReferenceStream(
        deriveNeatRacingCarNetworkChildSeed(FLEET_CONTROL_TEST_TUPLE_SEED, 2),
        2,
      );
      expect(sharedNetwork.draws).toEqual([
        carOneStream[0],
        carTwoStream[0],
        carOneStream[1],
        carTwoStream[1],
      ]);

      // Assert: the two cars really sit on DIFFERENT streams.
      expect(carOneStream[0]).not.toBe(carTwoStream[0]);
    });

    it('returns the adopted network stream exactly as it found it', () => {
      // Arrange: a donor network that ALREADY carries a stream (the eval
      // side seeded it before publishing it).
      const relay = createFleetControlTestRelay();
      const donorNetwork = createFleetControlTestStreamNetwork();
      donorNetwork.setSeed?.(2024);
      const donorState = donorNetwork.getRNGState?.();
      relay.publish(1, donorNetwork);
      const seam = createFleetControlTestSeam(relay.relay);
      const fleet = createFleetControlTestFleet();

      // Act: resolve one tick for the agent car.
      seam.computeControl(fleet, 1);

      // Assert: the donor's stream is byte-identical after the tick (the
      // seam BORROWED it and handed it back).
      expect(donorNetwork.getRNGState?.()).toBe(donorState);
    });
  });
});
