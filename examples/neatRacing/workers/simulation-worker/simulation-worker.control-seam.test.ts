/**
 * RED home for Step 23 (slice `P4-S1-red`) of `plans/neatracing-voxel-racer.plans.md`.
 *
 * FOUR contract groups live here — one `describe` per RED-P4 contract, and
 * inside each `describe` ONE `it` per single-purpose behavior:
 * - RED-P4-1 (AC-029): the ai-mode fleet runs the INJECTED NGE controller for
 *   cars 1..n, those inputs are DISTINCT from the scripted spline autopilot
 *   output for the SAME state, and car 0 keeps the replayed HUMAN sample in
 *   BOTH race modes (LD-12 lane isolation).
 * - RED-P4-2 (AC-043): the sim lane follows the CURRENTLY adopted network
 *   snapshot, ONE adopted-snapshot update round-trips into the sim lane, and
 *   a PRODUCTION module OUTSIDE the eval-worker subtree boots the eval worker
 *   session.
 * - RED-P4-3 (AC-043 / GAP-2): the goal-machine crossing feed carries a LIVE
 *   node count — the retired honesty constant
 *   `NEATRACING_DEMO_EVOLVED_NODE_COUNT` is GONE and NO numeric honesty
 *   constant replaces it.
 * - AC-026 PRESERVED (slice `P4-S2-concurrent-play`): the FIVE-CAR fleet
 *   replays byte-identically across the adaptation window WHILE the live
 *   adaptation + rival-swap streams run, and the worker boundary copies the
 *   seam's reused control buffer once per agent car per tick (no per-frame
 *   allocation growth against the one-car baseline).
 *
 * EVERY red in this file is a NAMED RUNTIME assertion failure BY
 * CONSTRUCTION: the file imports ONLY modules that already exist (at
 * RUNTIME), and every not-yet-existing seam is read through a FILE-LOCAL
 * WIDENED VIEW (the `as unknown as { ... }` idiom of
 * `physics.step.service.test.ts`), so a missing seam can NEVER surface as an
 * import failure, a compile error, or a "cannot find module" — it surfaces as
 * a legible assertion (or, for the resolver itself, a descriptive `Error`).
 *
 * The injected seam double is a CALLABLE PROXY: it answers EVERY plausible
 * seam shape (network / controller / per-car fleet-control / snapshot
 * carrier) with the SAME adopted activations, so the RED contracts pin the
 * OBSERVABLE control stream and never the exact production accessor name or
 * the per-car seeding strategy.
 *
 * The fixture is DETERMINISTIC: fixed seeds (42 / 43 / 44), the SHIPPED
 * autopilot fixture track (one straight closed loop of 40 lane-center
 * samples), NO RNG, NO timers, NO wall-clock reads.
 *
 * The GREEN side must make cars 1..n follow the injected controller seam
 * while car 0 keeps the replayed human sample in BOTH race modes.
 */

import { afterAll, beforeAll, describe, expect, it } from '@jest/globals';
import { readFileSync, readdirSync } from 'node:fs';
import { relative, resolve, sep } from 'node:path';

import { stepAutopilotInput } from '../../physics/ai.autopilot';
import { createTrackSpecForAutopilotFixture } from '../../physics/ai.autopilot.fixtures';
import { createInitialPhysicsCarState } from '../../physics/physics.step.service';
import type {
  PhysicsCarState,
  PhysicsStepInput,
} from '../../physics/physics.step.service';
import type { SplineSample } from '../../../racing_curriculum/track/track.generator.types';
import { NEATRACING_REPLAY_ADAPTATION_TICKS } from './reproducibility.utils';
import type { NeatRacingSimWorkerRaceMode } from './simulation-worker';

/** Lazily-loaded worker module type (imported in `beforeAll` after the stub). */
type SimWorkerModule = typeof import('./simulation-worker');

/**
 * Fake DedicatedWorkerGlobalScope the sim-worker module binds its message
 * adapter to at IMPORT time.
 */
const simWorkerScope: { onmessage: unknown; postMessage: jest.Mock } = {
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
  value: simWorkerScope,
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
let simWorkerModule: SimWorkerModule;

beforeAll(async () => {
  simWorkerModule = await import('./simulation-worker');
});

/**
 * The fixture spline samples: the SHIPPED autopilot fixture (one straight
 * closed loop of 40 lane-center samples along +x at y = 0, stride 1 m).
 *
 * The shipped fixture is reused on purpose: it drives the scripted autopilot
 * onto a SANE, non-degenerate racing line. A degenerate autopilot output
 * (a full-lock, zero-throttle answer) would make the "distinct from the
 * autopilot" assertion VACUOUS — nearly any control would differ from it —
 * and would weaken the RED contract for the agent-car path.
 */
const CONTROL_SEAM_TEST_SPLINE_SAMPLES: readonly SplineSample[] =
  createTrackSpecForAutopilotFixture();

/**
 * Builds the fixture fleet: car 0 is the HUMAN car, cars 1 and 2 are the
 * AGENT cars the controller seam must drive. All three sit ON the fixture
 * racing line (y = 0, heading +x) so the autopilot has a sane line to hold.
 *
 * @returns The three deterministic fixture car states.
 */
function createControlSeamTestFixtureFleet(): PhysicsCarState[] {
  return [
    createInitialPhysicsCarState(42, { x: 2, y: 0, heading: 0 }),
    createInitialPhysicsCarState(43, { x: 14, y: 0, heading: 0 }),
    createInitialPhysicsCarState(44, { x: 26, y: 0, heading: 0 }),
  ];
}

/**
 * The WIDENED view of the injected stub NGE controller seam.
 *
 * The stub answers EVERY plausible seam SHAPE so the RED contract does NOT
 * over-constrain the production wiring:
 * - NETWORK shape (`NeatRacingControllerNetwork`): `activate(vector)` answers
 *   the adopted network's output activations.
 * - CONTROLLER shape (`NeatRacingNgeController`): `computeControl(...)` and
 *   `computeControlWithEvidence(...)` answer the mapped control.
 * - FLEET-CONTROL shape (the per-car seam module): every other accessor
 *   answers the mapped control.
 * - SNAPSHOT-CARRIER shape: `activations` / `network` / `snapshot` expose the
 *   CURRENTLY adopted network.
 *
 * The stub's activations are deliberately SEED-INDEPENDENT: a production seam
 * that `setSeed()`s each car network from the child-seed table (the
 * `P4-S1-worker-seam` acceptance criterion) still observes the SAME adopted
 * activations for every car, so the RED contract pins the adopted snapshot
 * and NOT the per-car seeding strategy.
 */
type StubFleetController = {
  activate(vector: readonly number[]): readonly number[];
  computeControl(envState?: unknown, trackSpec?: unknown): PhysicsStepInput;
  computeControlWithEvidence(
    envState?: unknown,
    trackSpec?: unknown,
  ): { control: PhysicsStepInput; evidence: unknown };
  setSeed(seed: number): void;
  adopt(network: StubAdoptedNetwork): void;
  readonly nodeCount: number;
  readonly activations: readonly number[];
  (car?: unknown, carIndex?: number): PhysicsStepInput;
};

/**
 * The stub's output channel count: the NGE controller's two control axes
 * (`NEATRACING_NGE_CONTROLLER_OUTPUT_COUNT` — throttle, steer).
 */
const CONTROL_SEAM_STUB_OUTPUT_COUNT = 2;

/**
 * The accessor-name pattern that marks a SEAM UPDATE channel (an adopted
 * snapshot pushed INTO the seam) as opposed to a read accessor.
 */
const CONTROL_SEAM_SNAPSHOT_UPDATE_PATTERN =
  /(adopt|snapshot|network|update|commit|publish|push|replace|install|set)/i;

/**
 * The stub's ANSWER: an ACTIVATIONS ARRAY that ALSO carries the named control
 * axes.
 *
 * The hybrid shape keeps the RED contract SHAPE-AGNOSTIC: a production seam
 * that reads the answer as an activation vector (`answer[0]`, `answer[1]`) and
 * a production seam that reads it as a control input (`answer.throttle`,
 * `answer.steer`, `answer.brake`) observe the SAME values.
 */
type StubControlAnswer = number[] & PhysicsStepInput;

/**
 * Builds the stub answer for the given activations.
 *
 * Educational note: the derivation mirrors the shipped NGE controller law
 * (`ai/nge.controller.ts`): the throttle channel clamps into `[0, 1]`, the
 * steer channel clamps into `[-1, 1]`, the brake channel is always zero, and
 * NO turn-rate channel is ever emitted.
 *
 * @param activations - The adopted network's output activations.
 * @returns The hybrid activation/control answer.
 */
function createStubControlAnswer(
  activations: readonly number[],
): StubControlAnswer {
  const throttle = Math.max(0, Math.min(1, activations[0] ?? 0));
  const steer = Math.max(-1, Math.min(1, activations[1] ?? 0));
  const answer = [throttle, steer] as unknown as StubControlAnswer;
  answer.throttle = throttle;
  answer.brake = 0;
  answer.steer = steer;
  return answer;
}

/**
 * Reads the activations out of a SNAPSHOT-LIKE value (an activation array, an
 * object carrying `activations` / `outputs`, or a network-like object).
 *
 * @param candidate - The candidate snapshot value.
 * @returns The activations, or `null` when the value carries none.
 */
function readStubSnapshotActivations(
  candidate: unknown,
): readonly number[] | null {
  if (Array.isArray(candidate)) {
    const numeric = candidate.filter(
      (entry): entry is number => typeof entry === 'number',
    );
    return numeric.length === candidate.length && numeric.length > 0
      ? numeric
      : null;
  }
  if (typeof candidate !== 'object' || candidate === null) {
    return null;
  }
  const record = candidate as {
    activations?: unknown;
    outputs?: unknown;
    network?: { activations?: unknown } | null;
  };
  const candidates = [
    record.activations,
    record.outputs,
    record.network?.activations ?? null,
  ];
  for (const entry of candidates) {
    const activations = readStubSnapshotActivations(entry);
    if (activations !== null) {
      return activations;
    }
  }
  return null;
}

/**
 * The WIDENED view of the fleet tick-input resolver: the not-yet-existing
 * FIFTH parameter is the injected controller.
 */
type WidenedFleetTickResolver = (
  cars: readonly PhysicsCarState[],
  raceMode: NeatRacingSimWorkerRaceMode,
  humanSample: PhysicsStepInput | undefined,
  splineSamples: readonly SplineSample[],
  controller?: StubFleetController,
) => PhysicsStepInput[];

/**
 * The stub ADOPTED NETWORK double: the frozen snapshot the injected stub
 * controller currently holds.
 */
type StubAdoptedNetwork = {
  /** The network's output activations (throttle channel, steer channel). */
  readonly activations: readonly number[];
};

/** Adopted network A — the FIRST snapshot the sim lane must follow. */
const CONTROL_SEAM_NETWORK_A: StubAdoptedNetwork = {
  activations: [0.31, -0.73],
};

/** Adopted network B — the SECOND snapshot (the one adopted update). */
const CONTROL_SEAM_NETWORK_B: StubAdoptedNetwork = {
  activations: [0.62, 0.44],
};

/**
 * Creates the injected stub NGE controller double.
 *
 * Educational note: the double is a CALLABLE PROXY that answers EVERY
 * plausible seam accessor name — and EVERY name it does not know — with the
 * SAME derived control, so the RED contract constrains the OBSERVABLE control
 * stream and never the exact production accessor. A production seam that
 * treats the double as a network (`activate`), as a controller
 * (`computeControl`), or as a per-car fleet-control module (any other
 * accessor) all observe the same adopted activations.
 *
 * The double is also the SEAM's adopted-snapshot SOURCE: an update pushed
 * through any snapshot-update accessor (`adopt`, `setSnapshot`, ...) replaces
 * the adopted activations, and every later read observes the NEW snapshot.
 *
 * @param initialNetwork - The initially adopted network snapshot.
 * @returns The callable stub controller double.
 */
function createStubFleetController(
  initialNetwork: StubAdoptedNetwork,
): StubFleetController {
  let adoptedActivations: readonly number[] = initialNetwork.activations;

  const adoptSnapshot = (candidate: unknown): void => {
    const activations = readStubSnapshotActivations(candidate);
    if (activations !== null) {
      adoptedActivations = activations;
    }
  };

  const answer = (): StubControlAnswer =>
    createStubControlAnswer(adoptedActivations);

  // The double is CALLABLE: a production seam may invoke the seam directly
  // (`seam(car, carIndex)`) instead of calling a named accessor.
  const target = function stubControlSeam(...args: unknown[]): unknown {
    if (args.length === 1) {
      const activations = readStubSnapshotActivations(args[0]);
      if (activations !== null) {
        adoptedActivations = activations;
        return undefined;
      }
    }
    return answer();
  } as unknown as Record<string, unknown>;

  target.activate = (): StubControlAnswer => answer();
  target.computeControl = (): StubControlAnswer => answer();
  target.computeControlWithEvidence = (): {
    control: StubControlAnswer;
    evidence: unknown;
  } => ({
    control: answer(),
    evidence: { stubAdoptedActivations: adoptedActivations },
  });
  // The per-car child seed NEVER changes the stub's activations: the RED
  // contract pins the ADOPTED SNAPSHOT, not the per-car seeding strategy.
  target.setSeed = (): void => undefined;
  target.adopt = (candidate: unknown): void => {
    adoptSnapshot(candidate);
  };

  const proxy = new Proxy(target, {
    get: (source, property): unknown => {
      if (typeof property === 'symbol') {
        return undefined;
      }
      // A `then` accessor would make the double a THENABLE and would hang
      // any `await` on the seam.
      if (
        property === 'then' ||
        property === 'constructor' ||
        property === 'prototype' ||
        property === 'arguments' ||
        property === 'caller'
      ) {
        return undefined;
      }
      if (
        property === 'nodeCount' ||
        property === 'outputCount' ||
        property === 'length'
      ) {
        return CONTROL_SEAM_STUB_OUTPUT_COUNT;
      }
      if (property === 'activations' || property === 'outputs') {
        return adoptedActivations;
      }
      if (
        property === 'network' ||
        property === 'controller' ||
        property === 'seam' ||
        property === 'self'
      ) {
        return proxy;
      }
      if (
        property === 'snapshot' ||
        property === 'adopted' ||
        property === 'adoptedNetwork'
      ) {
        return {
          activations: adoptedActivations,
          nodeCount: CONTROL_SEAM_STUB_OUTPUT_COUNT,
        };
      }
      if (property in source) {
        return source[property];
      }
      // EVERY unknown accessor answers with the derived control, UNLESS its
      // name marks it as a snapshot-update channel.
      return (...args: unknown[]): unknown => {
        if (CONTROL_SEAM_SNAPSHOT_UPDATE_PATTERN.test(property)) {
          adoptSnapshot(args[0]);
          return undefined;
        }
        return answer();
      };
    },
    set: (source, property, value): boolean => {
      if (typeof property === 'symbol') {
        return true;
      }
      if (CONTROL_SEAM_SNAPSHOT_UPDATE_PATTERN.test(property)) {
        adoptSnapshot(value);
        return true;
      }
      source[property] = value;
      return true;
    },
    has: (): boolean => true,
  });

  return proxy as unknown as StubFleetController;
}

/**
 * Reads the fleet tick-input resolver through the FILE-LOCAL WIDENED VIEW
 * (the not-yet-widened FIFTH parameter is the injected controller).
 *
 * The member EXISTS today, so the descriptive throw never fires in the RED
 * phase; it only guards the not-yet-widened signature so a future rename
 * surfaces as a legible message instead of a confusing `TypeError`.
 *
 * @returns The widened fleet tick-input resolver.
 */
function resolveFleetTickInputs(): WidenedFleetTickResolver {
  const widenedInternals =
    simWorkerModule.NEATRACING_SIMULATION_WORKER_INTERNALS as unknown as {
      resolveSimWorkerFleetTickInputs?: WidenedFleetTickResolver;
    };
  const resolver = widenedInternals.resolveSimWorkerFleetTickInputs;
  if (resolver === undefined) {
    throw new Error(
      'simulation worker: resolveSimWorkerFleetTickInputs is missing',
    );
  }
  return resolver;
}

/**
 * Projects one control input onto the two float-safe channels the contracts
 * compare (the optional turn-rate channel is IGNORED on purpose: the NGE
 * seam never emits it, while the scripted autopilot does).
 *
 * @param input - The control input to project.
 * @returns The rounded throttle/steer pair.
 */
const projectControl = (
  input: PhysicsStepInput,
): { throttle: number; steer: number } => ({
  throttle: Number(input.throttle.toFixed(6)),
  steer: Number(input.steer.toFixed(6)),
});

/**
 * Collects the PRODUCTION sources OUTSIDE the eval-worker subtree that boot
 * the eval worker session (`createEvalWorkerSession(`).
 *
 * The scan is DETERMINISTIC: the collected repo-relative paths are sorted
 * before they are returned, so the assertion never depends on the
 * filesystem's directory order.
 *
 * @returns The sorted repo-relative paths of the host boot sources.
 */
function collectEvalWorkerBootSources(): readonly string[] {
  const root = resolve(process.cwd(), 'examples', 'neatRacing');
  const collected: string[] = [];

  const walk = (directory: string): void => {
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      const entryPath = resolve(directory, entry.name);
      if (entry.isDirectory()) {
        walk(entryPath);
        continue;
      }
      if (!entry.isFile() || !entry.name.endsWith('.ts')) {
        continue;
      }
      if (entry.name.endsWith('.test.ts')) {
        continue;
      }
      const relativePath = relative(root, entryPath).split(sep).join('/');
      if (relativePath.startsWith('workers/eval-worker/')) {
        continue;
      }
      const source = readFileSync(entryPath, 'utf-8');
      if (!source.includes('createEvalWorkerSession(')) {
        continue;
      }
      collected.push(relativePath);
    }
  };

  walk(root);
  return collected.sort();
}

describe('RED-P4-1 - the ai-mode fleet runs the injected NGE controller (AC-029)', () => {
  it('drives the agent cars through the injected controller, off the scripted autopilot', () => {
    // Arrange: the deterministic fixture spline samples, the three-car
    // fleet, the injected stub controller (adopted network A) and the human
    // sample.
    const cars = createControlSeamTestFixtureFleet();
    const controller = createStubFleetController(CONTROL_SEAM_NETWORK_A);
    const humanSample: PhysicsStepInput = {
      throttle: 1,
      brake: 0,
      steer: -0.5,
    };

    // Act: resolve one tick's inputs in ai mode with the injected
    // controller passed through the seam.
    const resolveFleetTickInputsOf = resolveFleetTickInputs();
    const aiInputs = resolveFleetTickInputsOf(
      cars,
      'ai',
      humanSample,
      CONTROL_SEAM_TEST_SPLINE_SAMPLES,
      controller,
    );

    // Assert: the AGENT cars follow the injected controller's activations.
    expect(aiInputs.slice(1).map(projectControl)).toEqual([
      { throttle: 0.31, steer: -0.73 },
      { throttle: 0.31, steer: -0.73 },
    ]);

    // Assert: those inputs are DISTINCT from the scripted spline autopilot
    // output for the SAME state (the agent path left the autopilot).
    const autopilotProjection = cars.map((car) =>
      projectControl(stepAutopilotInput(car, CONTROL_SEAM_TEST_SPLINE_SAMPLES)),
    );
    expect(aiInputs.slice(1).map(projectControl)).not.toEqual(
      autopilotProjection.slice(1),
    );
  });

  it('keeps car 0 on the replayed human sample in BOTH race modes (LD-12 lane isolation)', () => {
    // Arrange: the deterministic fixture spline samples, the three-car
    // fleet, the injected stub controller and the human sample.
    const cars = createControlSeamTestFixtureFleet();
    const controller = createStubFleetController(CONTROL_SEAM_NETWORK_A);
    const humanSample: PhysicsStepInput = {
      throttle: 1,
      brake: 0,
      steer: -0.5,
    };

    // Act: resolve one tick's inputs in ai mode AND in manual mode, both
    // with the injected controller passed through the seam.
    const resolveFleetTickInputsOf = resolveFleetTickInputs();
    const aiInputs = resolveFleetTickInputsOf(
      cars,
      'ai',
      humanSample,
      CONTROL_SEAM_TEST_SPLINE_SAMPLES,
      controller,
    );
    const manualInputs = resolveFleetTickInputsOf(
      cars,
      'manual',
      humanSample,
      CONTROL_SEAM_TEST_SPLINE_SAMPLES,
      controller,
    );

    // Assert: car 0 keeps the replayed HUMAN sample in BOTH race modes.
    expect([
      projectControl(aiInputs[0]),
      projectControl(manualInputs[0]),
    ]).toEqual([
      { throttle: 1, steer: -0.5 },
      { throttle: 1, steer: -0.5 },
    ]);
  });
});

describe('RED-P4-2 - the host adaptation session boots the eval worker and relays the adopted snapshot (AC-043)', () => {
  it('round-trips ONE adopted-snapshot update into the sim lane (no cached snapshot)', () => {
    // Arrange: the deterministic fixture spline samples, the three-car
    // fleet and the injected stub controller holding adopted network A.
    const cars = createControlSeamTestFixtureFleet();
    const controller = createStubFleetController(CONTROL_SEAM_NETWORK_A);
    const resolveFleetTickInputsOf = resolveFleetTickInputs();

    // Act: resolve one ai-mode tick, then push ONE adopted-snapshot update
    // through the seam and resolve again.
    const beforeUpdate = resolveFleetTickInputsOf(
      cars,
      'ai',
      undefined,
      CONTROL_SEAM_TEST_SPLINE_SAMPLES,
      controller,
    );
    controller.adopt(CONTROL_SEAM_NETWORK_B);
    const afterUpdate = resolveFleetTickInputsOf(
      cars,
      'ai',
      undefined,
      CONTROL_SEAM_TEST_SPLINE_SAMPLES,
      controller,
    );

    // Assert: the update ROUND-TRIPS into the sim lane — the lane follows the
    // LATEST adopted snapshot (a cached snapshot would stay on network A).
    expect(afterUpdate.slice(1).map(projectControl)).toEqual([
      { throttle: 0.62, steer: 0.44 },
      { throttle: 0.62, steer: 0.44 },
    ]);

    // Assert: the PRE-update lane really was on the OLDER snapshot, so the
    // round-trip is an observable delta and not a constant answer.
    expect(beforeUpdate.slice(1).map(projectControl)).toEqual([
      { throttle: 0.31, steer: -0.73 },
      { throttle: 0.31, steer: -0.73 },
    ]);
  });

  it('boots the eval worker session from a production module outside the eval-worker subtree (GAP-1 orphan consumption)', () => {
    // Act: scan the production sources for the eval-worker session boot.
    const bootSources = collectEvalWorkerBootSources();

    // Assert: a PRODUCTION module outside the eval-worker subtree boots the
    // eval worker session (today the ONLY boot is the eval worker's own
    // module-side-effect boot — the orphan-consumption gap).
    expect(bootSources.length).toBeGreaterThan(0);
  });
});

describe('RED-P4-3 - the goal-machine crossing feed carries a live nodeCount (AC-043 / GAP-2)', () => {
  it('drops the retired honesty constant, adds no numeric replacement, and keeps carrying a node count', () => {
    // Arrange: the goal-crossing feed source.
    const source = readFileSync(
      resolve(
        process.cwd(),
        'examples',
        'neatRacing',
        'browser-entry',
        'browser-entry.ts',
      ),
      'utf-8',
    );

    // Assert: the retired honesty constant is GONE from the feed source.
    expect(source).not.toContain('NEATRACING_DEMO_EVOLVED_NODE_COUNT');

    // Assert: NO numeric honesty constant replaces it. Only the honesty
    // constant family is banned — a LIVE rung threshold such as
    // `PROMOTION_NODE_COUNT_THRESHOLD` stays legal.
    expect(source).not.toMatch(
      /const\s+(?:[A-Z0-9_]*EVOLVED[A-Z0-9_]*|NEATRACING_DEMO_[A-Z0-9_]*)NODE_COUNT[A-Z0-9_]*\s*=\s*\d/,
    );

    // Assert: the crossing feed still carries a node count (the feed is not
    // gutted — the LIVE crossing node count replaces the constant).
    expect(source).toMatch(/\bnodeCount\b/);
  });
});

/* ------------------------------------------------------------------------- *
 * AC-026 PRESERVED — the five-car fleet under the live adaptation + the
 * rival-swap streams (slice `P4-S2-concurrent-play`).
 *
 * Educational note — WHAT THIS BLOCK OWNS: the byte-frozen RED home
 * (`host/promotion-live.red.test.ts`) owns the LIVE lane's five-car replay
 * (RED-P4-6). This block owns the SAME preservation law at the WORKER
 * BOUNDARY (the sim worker's fleet tick-input resolver):
 * - the five-car fleet replays BYTE-IDENTICALLY across the adaptation window
 *   WHILE the live adaptation stream and the rival-swap stream are running (a
 *   frozen snapshot would make the parity vacuous),
 * - both streams reach the sim lane: the adaptation stream moves EVERY agent
 *   car, the rival-swap stream moves ONLY the swapped rival,
 * - the worker boundary copies the seam's REUSED control buffer exactly once
 *   per agent car per tick (no per-frame allocation growth against the
 *   one-car baseline).
 *
 * The fixture is DETERMINISTIC: fixed per-car snapshots, a fixed tick
 * schedule, the SHIPPED autopilot fixture track, NO RNG, NO timers, NO
 * wall-clock reads.
 * ------------------------------------------------------------------------- */

/** The five-car fleet size the level-3 rung materializes (car 0 + 4 rivals). */
const CONTROL_SEAM_FIVE_CAR_COUNT = 5;

/** The one-car baseline fleet size (the level-1 rung: the human car only). */
const CONTROL_SEAM_ONE_CAR_COUNT = 1;

/** The fleet indices of the five-car fleet's AGENT cars (car 0 is human). */
const CONTROL_SEAM_AGENT_CAR_INDICES: readonly number[] = [1, 2, 3, 4];

/** The fleet index of the RIVAL whose brain the rival-swap stream replaces. */
const CONTROL_SEAM_RIVAL_SWAP_CAR_INDEX = 3;

/** The tick at which the live adaptation stream publishes fresh snapshots. */
const CONTROL_SEAM_ADAPTATION_STREAM_TICK = 8;

/** The tick at which the rival-swap stream swaps the rival's brain in. */
const CONTROL_SEAM_RIVAL_SWAP_STREAM_TICK = 16;

/** The car index the double answers for when the call names no car. */
const CONTROL_SEAM_DEFAULT_CAR_INDEX = 1;

/** The activations a car without an adopted snapshot is answered with. */
const CONTROL_SEAM_DEFAULT_ACTIVATIONS: readonly number[] = [0, 0];

/** The brake channel the double always publishes (the seam never brakes). */
const CONTROL_SEAM_STUB_BRAKE = 0;

/** The decimals the replay projection keeps (the byte-parity precision). */
const CONTROL_SEAM_PROJECTION_DECIMALS = 6;

/** The meters one tick advances a car per unit of published throttle. */
const CONTROL_SEAM_TEST_ADVANCE_M_PER_THROTTLE = 0.5;

/** The meters-per-second a car's speed follows its published throttle. */
const CONTROL_SEAM_TEST_SPEED_PER_THROTTLE = 10;

/**
 * The five-car fleet's initially adopted snapshots (indexed by car index).
 *
 * Educational note: the HUMAN car (index 0) owns NO adopted network — the
 * lane-isolation law — so its slot stays empty.
 */
const CONTROL_SEAM_FIVE_CAR_NETWORKS: readonly (
  StubAdoptedNetwork | undefined
)[] = [
  undefined,
  { activations: [0.31, -0.73] },
  { activations: [0.52, 0.18] },
  { activations: [0.73, -0.41] },
  { activations: [0.24, 0.65] },
];

/**
 * The snapshots the live adaptation stream publishes for the agent cars
 * (entry `i` belongs to car `i + 1`).
 */
const CONTROL_SEAM_ADOPTED_NETWORKS: readonly StubAdoptedNetwork[] = [
  { activations: [0.84, 0.27] },
  { activations: [0.19, -0.62] },
  { activations: [0.47, 0.83] },
  { activations: [0.66, -0.35] },
];

/**
 * The brain the rival-swap stream swaps INTO the rival car.
 */
const CONTROL_SEAM_SWAPPED_RIVAL_NETWORK: StubAdoptedNetwork = {
  activations: [0.91, -0.19],
};

/**
 * Creates the REUSED published control buffer of one car.
 *
 * Educational note — WHY A REUSED BUFFER: the shipped fleet-control seam
 * publishes a REUSED per-car buffer (the zero-allocation law), so a consumer
 * that HOLDS a published control across ticks must COPY it. The double
 * mirrors that law, which is what makes the worker boundary's copy law
 * observable (a worker that ALIASED the buffer would publish the SAME
 * instance on every tick).
 *
 * Educational note — THE CHANNELS: the buffer carries the NGE control axes
 * in the shipped order (channel 0 = throttle, channel 1 = steer — the
 * `mapNeatRacingFleetControl` mapping law), and the named axes carry the same
 * numbers.
 *
 * @returns The fresh (empty) reusable buffer.
 */
function createReusableStubControlAnswer(): StubControlAnswer {
  const answer = [0, 0] as unknown as StubControlAnswer;
  answer.throttle = 0;
  answer.brake = CONTROL_SEAM_STUB_BRAKE;
  answer.steer = 0;
  return answer;
}

/**
 * The per-car stub NGE controller: the five-car fleet's LIVE per-car adopted
 * snapshots.
 *
 * Educational note — WHY A PER-CAR DOUBLE: the fleet-control seam drives
 * every agent car through its OWN adopted network (the §9.7.2 `car-<i>`
 * table), so the AC-026 window needs a double that holds a DIFFERENT snapshot
 * per car and can REPLACE one car's snapshot mid-window (the live adaptation
 * stream and the rival-swap stream).
 *
 * The double answers the SAME plausible seam shapes as the single-snapshot
 * stub above — it is a CALLABLE PROXY — but it reads the car index out of the
 * call arguments, so a seam that asks for car `<i>`'s control observes car
 * `<i>`'s OWN adopted snapshot.
 */
type PerCarStubFleetController = StubFleetController & {
  /** Replaces ONE car's adopted snapshot (the live adaptation stream). */
  adoptForCar(carIndex: number, network: StubAdoptedNetwork): void;
};

/**
 * Creates the injected PER-CAR stub NGE controller double.
 *
 * @param initialNetworks - The initially adopted snapshot of every car
 *   (indexed by car index; `undefined` marks a car with no adopted network).
 * @returns The callable per-car stub controller double.
 */
function createPerCarStubFleetController(
  initialNetworks: readonly (StubAdoptedNetwork | undefined)[],
): PerCarStubFleetController {
  const adoptedByCar = new Map<number, StubAdoptedNetwork>();
  const publishedBufferByCar = new Map<number, StubControlAnswer>();
  initialNetworks.forEach((network, carIndex) => {
    if (network !== undefined) {
      adoptedByCar.set(carIndex, network);
    }
  });

  const readCarIndex = (candidate: unknown): number =>
    typeof candidate === 'number' && Number.isInteger(candidate)
      ? candidate
      : CONTROL_SEAM_DEFAULT_CAR_INDEX;

  const publishForCar = (carIndex: number): StubControlAnswer => {
    const activations =
      adoptedByCar.get(carIndex)?.activations ??
      CONTROL_SEAM_DEFAULT_ACTIVATIONS;
    const throttle = Math.max(0, Math.min(1, activations[0] ?? 0));
    const steer = Math.max(-1, Math.min(1, activations[1] ?? 0));
    const buffer =
      publishedBufferByCar.get(carIndex) ?? createReusableStubControlAnswer();
    publishedBufferByCar.set(carIndex, buffer);
    // The NGE control channels (channel 0 = throttle, channel 1 = steer).
    buffer[0] = throttle;
    buffer[1] = steer;
    buffer.throttle = throttle;
    buffer.brake = CONTROL_SEAM_STUB_BRAKE;
    buffer.steer = steer;
    return buffer;
  };

  const target = function stubControlSeam(...args: unknown[]): unknown {
    return publishForCar(readCarIndex(args[1] ?? args[0]));
  } as unknown as Record<string, unknown>;

  target.activate = (): StubControlAnswer =>
    publishForCar(CONTROL_SEAM_DEFAULT_CAR_INDEX);
  target.computeControl = (
    envState?: unknown,
    carIndex?: unknown,
  ): StubControlAnswer => publishForCar(readCarIndex(carIndex ?? envState));
  target.computeControlWithEvidence = (
    envState?: unknown,
    carIndex?: unknown,
  ): { control: StubControlAnswer; evidence: unknown } => {
    const resolvedCarIndex = readCarIndex(carIndex ?? envState);
    return {
      control: publishForCar(resolvedCarIndex),
      evidence: {
        stubAdoptedActivations:
          adoptedByCar.get(resolvedCarIndex)?.activations ?? null,
      },
    };
  };
  // The per-car child seed NEVER changes the double's activations: the
  // contract pins the ADOPTED SNAPSHOT, not the per-car seeding strategy.
  target.setSeed = (): void => undefined;
  target.adopt = (candidate: unknown): void => {
    const activations = readStubSnapshotActivations(candidate);
    if (activations !== null) {
      adoptedByCar.set(CONTROL_SEAM_DEFAULT_CAR_INDEX, { activations });
    }
  };
  target.adoptForCar = (carIndex: number, candidate: unknown): void => {
    const activations = readStubSnapshotActivations(candidate);
    if (activations !== null) {
      adoptedByCar.set(carIndex, { activations });
    }
  };

  const proxy = new Proxy(target, {
    get: (source, property): unknown => {
      if (typeof property === 'symbol') {
        return undefined;
      }
      // A `then` accessor would make the double a THENABLE and would hang
      // any `await` on the seam.
      if (
        property === 'then' ||
        property === 'constructor' ||
        property === 'prototype' ||
        property === 'arguments' ||
        property === 'caller'
      ) {
        return undefined;
      }
      if (
        property === 'nodeCount' ||
        property === 'outputCount' ||
        property === 'length'
      ) {
        return CONTROL_SEAM_STUB_OUTPUT_COUNT;
      }
      if (property === 'activations' || property === 'outputs') {
        return (
          adoptedByCar.get(CONTROL_SEAM_DEFAULT_CAR_INDEX)?.activations ??
          CONTROL_SEAM_DEFAULT_ACTIVATIONS
        );
      }
      if (
        property === 'network' ||
        property === 'controller' ||
        property === 'seam' ||
        property === 'self'
      ) {
        return proxy;
      }
      if (
        property === 'snapshot' ||
        property === 'adopted' ||
        property === 'adoptedNetwork'
      ) {
        return {
          activations:
            adoptedByCar.get(CONTROL_SEAM_DEFAULT_CAR_INDEX)?.activations ??
            CONTROL_SEAM_DEFAULT_ACTIVATIONS,
          nodeCount: CONTROL_SEAM_STUB_OUTPUT_COUNT,
        };
      }
      if (property in source) {
        return source[property];
      }
      // EVERY unknown accessor answers with the derived control, UNLESS its
      // name marks it as a snapshot-update channel.
      return (...args: unknown[]): unknown => {
        if (CONTROL_SEAM_SNAPSHOT_UPDATE_PATTERN.test(property)) {
          const activations = readStubSnapshotActivations(args[0]);
          if (activations !== null) {
            adoptedByCar.set(CONTROL_SEAM_DEFAULT_CAR_INDEX, { activations });
          }
          return undefined;
        }
        return publishForCar(readCarIndex(args[1] ?? args[0]));
      };
    },
    set: (source, property, value): boolean => {
      if (typeof property === 'symbol') {
        return true;
      }
      if (
        property !== 'adoptForCar' &&
        CONTROL_SEAM_SNAPSHOT_UPDATE_PATTERN.test(property)
      ) {
        const activations = readStubSnapshotActivations(value);
        if (activations !== null) {
          adoptedByCar.set(CONTROL_SEAM_DEFAULT_CAR_INDEX, { activations });
        }
        return true;
      }
      source[property] = value;
      return true;
    },
    has: (): boolean => true,
  });

  return proxy as unknown as PerCarStubFleetController;
}

/**
 * Builds the five-car fixture fleet: car 0 is the HUMAN car, cars 1..4 are
 * the AGENT cars (the level-3 rung's four rivals). Every car sits ON the
 * fixture racing line (y = 0, heading +x).
 *
 * @returns The five deterministic fixture car states.
 */
function createControlSeamTestFiveCarFleet(): PhysicsCarState[] {
  return [
    createInitialPhysicsCarState(42, { x: 2, y: 0, heading: 0 }),
    createInitialPhysicsCarState(43, { x: 10, y: 0, heading: 0 }),
    createInitialPhysicsCarState(44, { x: 18, y: 0, heading: 0 }),
    createInitialPhysicsCarState(45, { x: 26, y: 0, heading: 0 }),
    createInitialPhysicsCarState(46, { x: 34, y: 0, heading: 0 }),
  ];
}

/**
 * Builds the ONE-CAR baseline fleet (the level-1 rung: the human car only).
 *
 * @returns The single deterministic fixture car state.
 */
function createControlSeamTestOneCarFleet(): PhysicsCarState[] {
  return [createInitialPhysicsCarState(42, { x: 2, y: 0, heading: 0 })];
}

/**
 * Advances the fixture fleet by ONE tick of the published controls.
 *
 * @param fleet - The live fixture fleet (mutated in place).
 * @param controls - The published controls of this tick, one per car.
 */
function advanceControlSeamTestFleet(
  fleet: readonly PhysicsCarState[],
  controls: readonly PhysicsStepInput[],
): void {
  for (let carIndex = 0; carIndex < fleet.length; carIndex += 1) {
    const control = controls[carIndex];
    const car = fleet[carIndex];
    if (control === undefined || car === undefined) {
      continue;
    }
    car.x += control.throttle * CONTROL_SEAM_TEST_ADVANCE_M_PER_THROTTLE;
    car.speed = control.throttle * CONTROL_SEAM_TEST_SPEED_PER_THROTTLE;
  }
}

/**
 * Compares two projected controls BY VALUE.
 *
 * Educational note: the projection is a FRESH object per call, so two
 * projected controls must be compared by their numbers (an identity
 * comparison would always report a difference).
 *
 * @param first - The first projected control.
 * @param second - The second projected control.
 * @returns Whether the two controls carry the same numbers.
 */
function areControlProjectionsEqual(
  first: { throttle: number; steer: number } | undefined,
  second: { throttle: number; steer: number } | undefined,
): boolean {
  if (first === undefined || second === undefined) {
    return false;
  }
  return first.throttle === second.throttle && first.steer === second.steer;
}

/**
 * The per-tick record of one five-car replay window.
 */
type ControlSeamReplayTick = {
  /** The published control of every fleet car (the human slot included). */
  readonly controls: readonly { throttle: number; steer: number }[];
  /** The fleet positions after the tick (the fleet's own evolution). */
  readonly positions: readonly number[];
};

/**
 * Runs ONE five-car replay window of the AC-026 preservation scenario.
 *
 * Educational note — THE TWO LIVE STREAMS: the adaptation stream publishes a
 * FRESH snapshot for every agent car at its scheduled tick (the live
 * adaptation lane), and the rival-swap stream REPLACES one rival's brain at
 * its scheduled tick (the rival-swap lane). Both fire INSIDE the replay
 * window, so the byte-parity the window witnesses is the parity of a
 * CONCURRENTLY PLAYING fleet, not of a frozen one.
 *
 * @returns The per-tick record of the whole window.
 */
function runControlSeamFiveCarReplay(): readonly ControlSeamReplayTick[] {
  // Arrange: the five-car fleet, the per-car double and the human sample.
  const cars = createControlSeamTestFiveCarFleet();
  const controller = createPerCarStubFleetController(
    CONTROL_SEAM_FIVE_CAR_NETWORKS,
  );
  const resolveFleetTickInputsOf = resolveFleetTickInputs();
  const humanSample: PhysicsStepInput = { throttle: 1, brake: 0, steer: -0.5 };
  const ticks: ControlSeamReplayTick[] = [];

  // Act: drive the whole adaptation window.
  for (let tick = 0; tick < NEATRACING_REPLAY_ADAPTATION_TICKS; tick += 1) {
    // The live adaptation stream: every agent car adopts a FRESH snapshot.
    if (tick === CONTROL_SEAM_ADAPTATION_STREAM_TICK) {
      CONTROL_SEAM_ADOPTED_NETWORKS.forEach((network, carIndex) => {
        controller.adoptForCar(carIndex + 1, network);
      });
    }
    // The rival-swap stream: ONE rival's brain is REPLACED mid-window.
    if (tick === CONTROL_SEAM_RIVAL_SWAP_STREAM_TICK) {
      controller.adoptForCar(
        CONTROL_SEAM_RIVAL_SWAP_CAR_INDEX,
        CONTROL_SEAM_SWAPPED_RIVAL_NETWORK,
      );
    }

    const inputs = resolveFleetTickInputsOf(
      cars,
      'ai',
      humanSample,
      CONTROL_SEAM_TEST_SPLINE_SAMPLES,
      controller,
    );
    ticks.push({
      controls: inputs.map(projectControl),
      positions: cars.map((car) =>
        Number(car.x.toFixed(CONTROL_SEAM_PROJECTION_DECIMALS)),
      ),
    });
    advanceControlSeamTestFleet(cars, inputs);
  }

  return ticks;
}

/**
 * Measures the worker boundary's per-tick input facts of one fleet.
 *
 * Educational note — WHAT IS MEASURED: the worker boundary publishes ONE
 * input per fleet car per tick, and the AGENT cars' inputs are FRESH copies
 * (the seam's reused control buffer is copied at the boundary, so a worker
 * that ALIASED it would publish the SAME instance on every tick).
 *
 * @param fleet - The fleet to drive.
 * @returns The measured per-tick facts.
 */
function measureControlSeamTickInputs(fleet: readonly PhysicsCarState[]): {
  readonly perTickInputCounts: readonly number[];
  readonly agentInputsAreFreshCopies: boolean;
} {
  const controller = createPerCarStubFleetController(
    CONTROL_SEAM_FIVE_CAR_NETWORKS,
  );
  const resolveFleetTickInputsOf = resolveFleetTickInputs();
  const humanSample: PhysicsStepInput = { throttle: 1, brake: 0, steer: -0.5 };
  const perTickInputCounts: number[] = [];
  const agentInputs: PhysicsStepInput[] = [];

  for (let tick = 0; tick < NEATRACING_REPLAY_ADAPTATION_TICKS; tick += 1) {
    const inputs = resolveFleetTickInputsOf(
      fleet,
      'ai',
      humanSample,
      CONTROL_SEAM_TEST_SPLINE_SAMPLES,
      controller,
    );
    perTickInputCounts.push(inputs.length);
    for (let carIndex = 1; carIndex < inputs.length; carIndex += 1) {
      const input = inputs[carIndex];
      if (input !== undefined) {
        agentInputs.push(input);
      }
    }
  }

  return {
    perTickInputCounts,
    agentInputsAreFreshCopies: new Set(agentInputs).size === agentInputs.length,
  };
}

describe('AC-026 PRESERVED - the five-car fleet under the live adaptation + rival-swap streams (P4-S2-concurrent-play)', () => {
  it('replays the five-car fleet byte-identically across the adaptation window', () => {
    // Arrange + Act: two INDEPENDENT replays of the SAME five-car window.
    const streamA = runControlSeamFiveCarReplay();
    const streamB = runControlSeamFiveCarReplay();

    // Assert: both replays are byte-identical (the AC-026 preservation law
    // holds WHILE the live adaptation + rival-swap streams are running).
    expect(JSON.stringify(streamA)).toBe(JSON.stringify(streamB));
  });

  it('carries the live adaptation stream and the rival-swap stream into the sim lane', () => {
    // Arrange: one replay of the five-car window.
    const ticks = runControlSeamFiveCarReplay();

    // Act: read the ticks around the two stream ticks.
    const beforeAdaptation = ticks[CONTROL_SEAM_ADAPTATION_STREAM_TICK - 1];
    const afterAdaptation = ticks[CONTROL_SEAM_ADAPTATION_STREAM_TICK];
    const beforeSwap = ticks[CONTROL_SEAM_RIVAL_SWAP_STREAM_TICK - 1];
    const afterSwap = ticks[CONTROL_SEAM_RIVAL_SWAP_STREAM_TICK];
    if (
      beforeAdaptation === undefined ||
      afterAdaptation === undefined ||
      beforeSwap === undefined ||
      afterSwap === undefined
    ) {
      throw new Error(
        'the five-car replay window is shorter than its own stream ticks',
      );
    }

    // Assert: the live adaptation stream moved EVERY agent car, and the
    // rival-swap stream moved ONLY the swapped rival (the other rivals keep
    // their own brain).
    expect({
      adaptationStreamMovedEveryAgentCar: CONTROL_SEAM_AGENT_CAR_INDICES.every(
        (carIndex) =>
          !areControlProjectionsEqual(
            beforeAdaptation.controls[carIndex],
            afterAdaptation.controls[carIndex],
          ),
      ),
      rivalSwapMovedOnlyTheSwappedRival:
        !areControlProjectionsEqual(
          beforeSwap.controls[CONTROL_SEAM_RIVAL_SWAP_CAR_INDEX],
          afterSwap.controls[CONTROL_SEAM_RIVAL_SWAP_CAR_INDEX],
        ) &&
        CONTROL_SEAM_AGENT_CAR_INDICES.filter(
          (carIndex) => carIndex !== CONTROL_SEAM_RIVAL_SWAP_CAR_INDEX,
        ).every((carIndex) =>
          areControlProjectionsEqual(
            beforeSwap.controls[carIndex],
            afterSwap.controls[carIndex],
          ),
        ),
    }).toEqual({
      adaptationStreamMovedEveryAgentCar: true,
      rivalSwapMovedOnlyTheSwappedRival: true,
    });
  });

  it('copies the seam control once per agent car per tick: the per-frame allocation never grows against the one-car baseline', () => {
    // Arrange: the five-car fleet and the one-car baseline fleet.
    const fiveCar = measureControlSeamTickInputs(
      createControlSeamTestFiveCarFleet(),
    );
    const oneCar = measureControlSeamTickInputs(
      createControlSeamTestOneCarFleet(),
    );

    // Assert: the worker boundary publishes ONE input per fleet car on the
    // FIRST and on the LAST tick of the window (the per-frame allocation
    // never grows), the agent cars' inputs are FRESH copies of the seam's
    // reused buffer, and the one-car baseline owns no agent input at all.
    expect({
      fiveCarFirstTickInputCount: fiveCar.perTickInputCounts[0],
      fiveCarLastTickInputCount:
        fiveCar.perTickInputCounts[NEATRACING_REPLAY_ADAPTATION_TICKS - 1],
      fiveCarInputCountNeverGrows:
        new Set(fiveCar.perTickInputCounts).size === 1,
      fiveCarAgentInputsAreFreshCopies: fiveCar.agentInputsAreFreshCopies,
      oneCarFirstTickInputCount: oneCar.perTickInputCounts[0],
      oneCarLastTickInputCount:
        oneCar.perTickInputCounts[NEATRACING_REPLAY_ADAPTATION_TICKS - 1],
      oneCarInputCountNeverGrows: new Set(oneCar.perTickInputCounts).size === 1,
    }).toEqual({
      fiveCarFirstTickInputCount: CONTROL_SEAM_FIVE_CAR_COUNT,
      fiveCarLastTickInputCount: CONTROL_SEAM_FIVE_CAR_COUNT,
      fiveCarInputCountNeverGrows: true,
      fiveCarAgentInputsAreFreshCopies: true,
      oneCarFirstTickInputCount: CONTROL_SEAM_ONE_CAR_COUNT,
      oneCarLastTickInputCount: CONTROL_SEAM_ONE_CAR_COUNT,
      oneCarInputCountNeverGrows: true,
    });
  });
});
