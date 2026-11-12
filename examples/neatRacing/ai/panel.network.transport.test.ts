/** @jest-environment jsdom */

import { afterAll, beforeAll, describe, expect, it } from '@jest/globals';
import Network from '../../../src/architecture/network';
import type { NeatRacingNetworkPanelCandidate } from '../visualizer/network-panel.adapter';
import type { NeatRacingFleetControlSeam } from '../workers/simulation-worker/nge.fleet.control';
import type {
  PhysicsCarState,
  PhysicsStepInput,
} from '../physics/physics.step.service';
import type {
  NeatRacingSimWorkerInitPayload,
  NeatRacingSimWorkerRequest,
} from '../workers/simulation-worker/simulation-worker';
import { NEATRACING_NGE_CONTROLLER_OUTPUT_COUNT } from './nge.controller';
import { deriveNeatRacingCarNetworkChildSeed } from '../workers/simulation-worker/reproducibility.utils';
import { resolveNeatRacingObservationChannelCount } from './observation.assembler';

/**
 * RED contracts for the docked-right panel NETWORK TRANSPORT (slice
 * P4-S2C-red — Phase 4, Step 24C of plans/neatracing-voxel-racer.plans.md,
 * RED-P4-22 + RED-P4-23).
 *
 * Educational note — the 1b defect these contracts pin down. The boot panel
 * (F3) opens the network panel correctly, but the host feed hard-wires
 * `network: undefined` for every candidate (the recorded "network: undefined"
 * feed bug): the winner's LIVE network never reaches the docked-right shelf,
 * and the panel never moves past idle. This red home pins the MISSING
 * TRANSPORT that the implementing slice (P4-S2C-transport) must add:
 *
 * - RED-P4-22 — THE CARRIER + ENVELOPE LAWS: the sim-worker request type
 *   carries the elected car index, and the worker replies with the SAME
 *   envelope law the promotion pipeline already uses, byte-serialized for
 *   THAT car's network at answer time (request → `{ type: 'network',
 *   carIndex }` → ready `{ type: 'network', payload: { kind: 'panel-network',
 *   carIndex, channelCount, outputCount, weights } }` envelope). The request
 *   type does not exist today: the worker's default dispatch branch answers
 *   `{ type: 'error', payload: { message: 'sim worker: unknown message type
 *   network' } }` instead.
 * - RED-P4-23 — THE LIVE-FEED LAWS: the host-side transport helper
 *   (`resolveNeatRacingPanelFeedNetwork`) rebuilds the reply envelope into
 *   the worker's exact network (same channels, same positional weights) and
 *   feeds it ONLY for the elected candidate: a transport-pending winner
 *   stays idle, and a foreign-car reply NEVER network-fies a runner-up.
 *
 * RED-P4-SHAPE: the reds fail at HEAD for the RIGHT reason only. No worker
 * or host production byte is authored here; the worker's unknown-message
 * error reply and the missing `ai/panel.network.transport.ts` module are the
 * two missing-behaviour witnesses the implementing slice must replace.
 *
 * Educational note — the jsdom docblock. The worker module binds
 * `self.onmessage` at import time inside its module scope (the same reason
 * the sim-worker green home fakes a worker scope and dynamically imports the
 * module AFTER the stub binds). The stub is restored in `afterAll`.
 */

/** Lazily-loaded worker module type (imported in `beforeAll` after the stub). */
type SimWorkerModule =
  typeof import('../workers/simulation-worker/simulation-worker');

/** Session handle type for helper signatures (the internals state machine). */
type SimWorkerSession = ReturnType<
  SimWorkerModule['NEATRACING_SIMULATION_WORKER_INTERNALS']['createSimWorkerSession']
>;

/** The NEW reply envelope shape the transport slice must answer with. */
interface RedPanelNetworkReplyEnvelope {
  readonly type: string;
  readonly payload: {
    readonly kind?: string;
    readonly carIndex?: number;
    readonly channelCount?: number;
    readonly outputCount?: number;
    readonly weights?: string;
    /** The worker's error-path message (the red failure's naming aid). */
    readonly message?: string;
  };
}

/** Fake DedicatedWorkerGlobalScope the worker module's onmessage adapter binds to. */
const redTransportWorkerScope: { onmessage: unknown; postMessage: jest.Mock } =
  {
    onmessage: undefined,
    postMessage: jest.fn(),
  };

/** Original `self` property descriptor captured before the test fakes it. */
const redTransportOriginalSelfDescriptor = Object.getOwnPropertyDescriptor(
  globalThis,
  'self',
);

Object.defineProperty(globalThis, 'self', {
  configurable: true,
  value: redTransportWorkerScope,
});

afterAll(() => {
  if (redTransportOriginalSelfDescriptor === undefined) {
    Object.defineProperty(globalThis, 'self', {
      configurable: true,
      value: undefined,
    });
  } else {
    Object.defineProperty(
      globalThis,
      'self',
      redTransportOriginalSelfDescriptor,
    );
  }
});

/** Fixture run seed for both the worker boot and the spawn-mind mirror. */
const RED_TRANSPORT_TEST_SEED = 42;

/** Fixture fleet size: the level-2 rung boot (car 0 + exactly one rival). */
const RED_TRANSPORT_TEST_AGENT_COUNT = 2;

/** The ladder rung the boot snapshot selects (a level-2 rival ladder). */
const RED_TRANSPORT_TEST_LADDER_LEVEL = 2;

/** The elected rival the transport requests (the rung's non-zero team slot). */
const RED_TRANSPORT_TEST_RIVAL_CAR_INDEX = 1;

/** Level-2 observation channel count (the envelope's channelCount facet). */
const RED_TRANSPORT_LEVEL_CHANNEL_COUNT =
  resolveNeatRacingObservationChannelCount(RED_TRANSPORT_TEST_LADDER_LEVEL);

/** The self-describing payload kind the envelope must carry (the envelope law). */
const RED_TRANSPORT_REPLY_ENVELOPE_KIND = 'panel-network';

/** The NEW request type the transport slice must add. */
const RED_TRANSPORT_REQUEST_TYPE = 'network';

/**
 * The hex-weights shape law: byte-exact lowercase hex, two chars per byte,
 * no separators (the promotion-lane envelope encoding mirrored).
 */
const RED_TRANSPORT_HEX_WEIGHTS_PATTERN = /^[0-9a-f]+$/;

/** Fixture track width (meters) for every fixture segment. */
const RED_TRANSPORT_TEST_TRACK_WIDTH_M = 60;

/** Worker module loaded in `beforeAll` once the fake worker scope has bound. */
let redTransportSimWorkerModule: SimWorkerModule;

beforeAll(async () => {
  redTransportSimWorkerModule =
    await import('../workers/simulation-worker/simulation-worker');
});

/**
 * The deterministic control answer the agent cars publish in this fixture.
 *
 * Educational note — the frozen answer keeps the boot deterministic (no RNG,
 * no clock): the addressed car's own heading is folded onto the steer axis
 * exactly like the sim-worker green home's double.
 */
const RED_TRANSPORT_TEST_AGENT_CONTROL: Readonly<PhysicsStepInput> =
  Object.freeze({
    throttle: 1,
    brake: 0,
    steer: 0.25,
  });

/**
 * Creates the deterministic controller double the agent cars run.
 *
 * @returns A deterministic fleet-control seam double.
 */
function createRedTransportControllerDouble(): NeatRacingFleetControlSeam {
  return {
    computeControl(
      fleet: readonly PhysicsCarState[],
      carIndex: number,
    ): PhysicsStepInput {
      const addressedCar = fleet[carIndex];
      return {
        throttle: RED_TRANSPORT_TEST_AGENT_CONTROL.throttle,
        brake: RED_TRANSPORT_TEST_AGENT_CONTROL.brake,
        steer: RED_TRANSPORT_TEST_AGENT_CONTROL.steer + addressedCar.heading,
      };
    },
  };
}

/**
 * Builds the closed-loop square fixture track (exactly the sim-worker green
 * home's fixture geometry: four axis-aligned segments whose endpoints close
 * ((0,0) → (800,0) → (800,800) → (0,800) → (0,0)), one spline sample per
 * segment midpoint, two lanes).
 *
 * @returns A minimal deterministic track spec fixture.
 */
function createRedTransportFixtureTrackSpec() {
  const segments = [
    { startX: 0, startY: 0, endX: 800, endY: 0 },
    { startX: 800, startY: 0, endX: 800, endY: 800 },
    { startX: 800, startY: 800, endX: 0, endY: 800 },
    { startX: 0, startY: 800, endX: 0, endY: 0 },
  ].map((segment) => ({
    ...segment,
    width: RED_TRANSPORT_TEST_TRACK_WIDTH_M,
  }));
  const splineSamples = [
    { x: 400, y: 0 },
    { x: 800, y: 400 },
    { x: 400, y: 800 },
    { x: 0, y: 400 },
  ].map((sample, globalIndex) => ({
    ...sample,
    width: RED_TRANSPORT_TEST_TRACK_WIDTH_M,
    segmentIndex: globalIndex,
    sampleIndexWithinSegment: 0,
    globalIndex,
  }));
  return {
    seed: RED_TRANSPORT_TEST_SEED,
    layoutVersion: 1,
    sizeBucket: 'fixture',
    segments,
    splineSamples,
    laneCount: 2,
  };
}

/**
 * Builds the rival-ladder boot payload: the level-2 rung's snapshot plus the
 * two-car fleet (car 0 + the rival car the ladder spawns).
 *
 * @returns A complete init payload for the level-2 rival-ladder boot.
 */
function createRedTransportRivalLadderInitPayload(): NeatRacingSimWorkerInitPayload {
  return {
    seed: RED_TRANSPORT_TEST_SEED,
    agentCount: RED_TRANSPORT_TEST_AGENT_COUNT,
    packSchemaVersion: 'neatracing-packed-v1',
    opponentSnapshot: { level: RED_TRANSPORT_TEST_LADDER_LEVEL },
    trackId: 'fixture-square-loop',
    featureFlags: { aiAutopilot: true },
    trackSpec: createRedTransportFixtureTrackSpec(),
    raceMode: 'ai',
    controller: createRedTransportControllerDouble(),
  };
}

/**
 * Boots one deterministic rival-ladder session and guards the boot.
 *
 * Educational note — the boot guard is a FIXTURE contract, not a red
 * contract: no frames are driven (the transport request is frame-independent
 * — the spawn mind exists from boot). A failed boot throws the named guard
 * error so the red failure can never be mistaken for a boot regression.
 *
 * @returns The booted session (init sent, ready reply verified).
 */
function createRedTransportRivalLadderSession(): SimWorkerSession {
  if (!redTransportSimWorkerModule) {
    throw new Error('simulation-worker module not loaded in beforeAll');
  }
  const session =
    redTransportSimWorkerModule.NEATRACING_SIMULATION_WORKER_INTERNALS.createSimWorkerSession();
  const bootResponses = session.handleRequest({
    type: 'init',
    payload: createRedTransportRivalLadderInitPayload(),
  });
  const readyReply = bootResponses[
    bootResponses.length - 1
  ] as NeatRacingSimWorkerResponseLike;
  if (bootResponses.length !== 1 || readyReply.type !== 'ready') {
    throw new Error(
      'RED-P4-22/23 red fixture guard: the rival-ladder boot did not answer exactly one ready reply',
    );
  }
  return session;
}

/** Narrow reply shape for the boot guard (avoid importing reply types by name). */
type NeatRacingSimWorkerResponseLike = { type: string; payload: unknown };

/**
 * Builds the NEW transport request for one elected car index.
 *
 * @param carIndex - The elected car the request names.
 * @returns The cast request (the type does not exist on the union at HEAD).
 */
function redTransportRequestFor(carIndex: number): NeatRacingSimWorkerRequest {
  return {
    type: RED_TRANSPORT_REQUEST_TYPE,
    carIndex,
  } as unknown as NeatRacingSimWorkerRequest;
}

/**
 * Sends one transport request and returns the narrowed reply envelope.
 *
 * @param session - The booted rival-ladder session.
 * @param carIndex - The elected car the request names.
 * @returns The sole reply cast to the NEW envelope shape (unknown at HEAD).
 */
function sendRedTransportRequest(
  session: SimWorkerSession,
  carIndex: number,
): RedPanelNetworkReplyEnvelope {
  const responses = session.handleRequest(redTransportRequestFor(carIndex));
  const [reply] = responses as unknown as [RedPanelNetworkReplyEnvelope];
  return reply;
}

/**
 * Mirrors the worker's spawn-mind law on the test side for one rung/car.
 *
 * @param seed - The booted run seed.
 * @param level - The ladder rung (selects the observation channel count).
 * @param carIndex - The rival's fleet car index (the per-car child seed).
 * @returns The positional connection-weight array of the mirrored spawn mind.
 */
function mirrorSpawnMindWeights(
  seed: number,
  level: 1 | 2 | 3,
  carIndex: number,
): Float32Array {
  const spawnMind = new Network(
    resolveNeatRacingObservationChannelCount(level),
    NEATRACING_NGE_CONTROLLER_OUTPUT_COUNT,
    { seed: deriveNeatRacingCarNetworkChildSeed(seed, carIndex) },
  );
  return serializeNetworkWeights(spawnMind);
}

/**
 * Serializes one network's connection weights positionally (the ledger law:
 * `network.connections[i].weight` in connection order) — mirrored from the
 * worker's serializer.
 *
 * @param network - The network to serialize.
 * @returns The positional weight array.
 */
function serializeNetworkWeights(network: Network): Float32Array {
  const weights = new Float32Array(network.connections.length);
  for (let index = 0; index < weights.length; index += 1) {
    weights[index] = network.connections[index].weight;
  }
  return weights;
}

/**
 * Mirrors the envelope hex decoding on the test side (native byte order).
 *
 * @param hex - The hex envelope string to decode.
 * @returns The decoded weights.
 */
function decodeWeightsFromHex(hex: string): Float32Array {
  const bytes = new Uint8Array(hex.length / 2);
  for (let byteIndex = 0; byteIndex < bytes.length; byteIndex += 1) {
    bytes[byteIndex] = parseInt(
      hex.slice(byteIndex * 2, byteIndex * 2 + 2),
      16,
    );
  }
  return new Float32Array(bytes.buffer);
}

/** The rival's mirrored spawn-mind weights (the byte-exactness anchor). */
const RED_TRANSPORT_MIRRORED_RIVAL_SPAWN_WEIGHTS = mirrorSpawnMindWeights(
  RED_TRANSPORT_TEST_SEED,
  RED_TRANSPORT_TEST_LADDER_LEVEL,
  RED_TRANSPORT_TEST_RIVAL_CAR_INDEX,
);

/**
 * Loads the (not-yet-existing) host transport module behind the red gate.
 *
 * Educational note — the red gate. At HEAD the file
 * `examples/neatRacing/ai/panel.network.transport.ts` does not exist, so the
 * import throws and the gate answers the NAMED missing-module error. At
 * GREEN the import resolves and the two pinned exports are checked. The
 * file is NEVER statically imported (a static missing import would fail the
 * suite at compile time — the wrong red reason); the widened string
 * specifier keeps the dynamic import compilable both at HEAD and at GREEN.
 *
 * @returns The transport module surface once both exports exist.
 */
async function gateRedPanelNetworkTransport(): Promise<{
  restoreNeatRacingPanelNetworkFromReply: (payload: unknown) => {
    carIndex: number;
    network: Network;
  };
  resolveNeatRacingPanelFeedNetwork: (
    elected: NeatRacingNetworkPanelCandidate,
    resolution: { carIndex: number; network: Network } | undefined,
  ) => Network | undefined;
}> {
  // Arranged with a widened variable on purpose: the literal path does not
  // exist yet, and TS would reject the literal specifier at HEAD (TS 2307) —
  // a compile failure is NOT the red reason.
  const transportModulePath: string = './panel.network.transport';
  let transportModule: Record<string, unknown> | undefined;
  try {
    transportModule = (await import(transportModulePath)) as Record<
      string,
      unknown
    >;
  } catch {
    transportModule = undefined;
  }
  const hasRebuild =
    typeof transportModule?.restoreNeatRacingPanelNetworkFromReply ===
    'function';
  const hasFeed =
    typeof transportModule?.resolveNeatRacingPanelFeedNetwork === 'function';
  if (!transportModule || !hasRebuild || !hasFeed) {
    throw new Error(
      'RED-P4-22/23 red gate: the panel network transport module (the reply codec restoreNeatRacingPanelNetworkFromReply + the elected feed helper resolveNeatRacingPanelFeedNetwork) does not exist yet at examples/neatRacing/ai/panel.network.transport.ts',
    );
  }
  return transportModule as unknown as {
    restoreNeatRacingPanelNetworkFromReply: (payload: unknown) => {
      carIndex: number;
      network: Network;
    };
    resolveNeatRacingPanelFeedNetwork: (
      elected: NeatRacingNetworkPanelCandidate,
      resolution: { carIndex: number; network: Network } | undefined,
    ) => Network | undefined;
  };
}

describe('NeatRacing panel network transport (RED-P4-22/23)', () => {
  describe('RED-P4-22 — the worker transport carrier + envelope laws', () => {
    it('answers a transport request for the elected rival with the panel-network envelope of THAT car', () => {
      // Arrange: the rival ladder boot (the level-2 rung: the rival car 1
      // exists from boot; no frames are driven — the transport request is
      // frame-independent).
      const session = createRedTransportRivalLadderSession();

      // Act: the NEW request type — the elected car index rides the request.
      const reply = sendRedTransportRequest(
        session,
        RED_TRANSPORT_TEST_RIVAL_CAR_INDEX,
      );

      // Assert: the reply IS the new envelope (right reason naming: the
      // received facets carry the worker's current unknown-message error).
      expect({
        replyCountAtAnswer: 1,
        replyType: reply.type,
        envelopeKind: reply.payload.kind,
        carrierCarIndex: reply.payload.carIndex,
        weightsAreHex: RED_TRANSPORT_HEX_WEIGHTS_PATTERN.test(
          reply.payload.weights ?? '',
        ),
        unknownMessageFallback: reply.payload.message,
      }).toEqual({
        replyCountAtAnswer: 1,
        replyType: RED_TRANSPORT_REQUEST_TYPE,
        envelopeKind: RED_TRANSPORT_REPLY_ENVELOPE_KIND,
        carrierCarIndex: RED_TRANSPORT_TEST_RIVAL_CAR_INDEX,
        weightsAreHex: true,
        unknownMessageFallback: undefined,
      });
    });

    it('carries THAT car’s spawn mind in the envelope law (channel/output counts + byte-exact hex)', () => {
      // Arrange: the boot + one transport request for the rival car.
      const session = createRedTransportRivalLadderSession();

      // Act: the reply's envelope facets of THIS car's network.
      const reply = sendRedTransportRequest(
        session,
        RED_TRANSPORT_TEST_RIVAL_CAR_INDEX,
      );
      const decodedWeights = decodeWeightsFromHex(reply.payload.weights ?? '');

      // Assert: the channel/output counts name THIS rung, and the weights
      // are byte-identical to the mirrored spawn mind of THAT car.
      expect(reply.payload.channelCount).toBe(
        RED_TRANSPORT_LEVEL_CHANNEL_COUNT,
      );
      expect(reply.payload.outputCount).toBe(
        NEATRACING_NGE_CONTROLLER_OUTPUT_COUNT,
      );
      expect(decodedWeights).toEqual(
        RED_TRANSPORT_MIRRORED_RIVAL_SPAWN_WEIGHTS,
      );
    });

    it('replies a COPY: two identical requests decode identical yet distinct weight arrays', () => {
      // Arrange: the boot + two identical transport requests.
      const session = createRedTransportRivalLadderSession();
      const firstReply = sendRedTransportRequest(
        session,
        RED_TRANSPORT_TEST_RIVAL_CAR_INDEX,
      );
      const secondReply = sendRedTransportRequest(
        session,
        RED_TRANSPORT_TEST_RIVAL_CAR_INDEX,
      );
      const firstWeights = decodeWeightsFromHex(
        firstReply.payload.weights ?? '',
      );
      const secondWeights = decodeWeightsFromHex(
        secondReply.payload.weights ?? '',
      );

      // Assert: identical bytes, DISTINCT decoded arrays (the reply is a
      // copy per request, never a shared ledger reference), and the ledger
      // keeps its spawn mind unchanged (no transport mutation).
      expect(firstWeights).toEqual(secondWeights);
      expect(firstWeights).not.toBe(secondWeights);
      expect(firstWeights).toEqual(RED_TRANSPORT_MIRRORED_RIVAL_SPAWN_WEIGHTS);
    });

    it('rebuilds the reply envelope idempotently into the same channels/weights network', async () => {
      // Arrange: the booted session, one real request, and the host-side
      // rebuild entry point (does not exist yet — the gate names it).
      const transport = await gateRedPanelNetworkTransport();
      const session = createRedTransportRivalLadderSession();
      const reply = sendRedTransportRequest(
        session,
        RED_TRANSPORT_TEST_RIVAL_CAR_INDEX,
      );

      // Act: the idempotent rebuild (twice, same reply payload).
      const firstRebuilt = transport.restoreNeatRacingPanelNetworkFromReply(
        reply.payload,
      );
      const secondRebuilt = transport.restoreNeatRacingPanelNetworkFromReply(
        reply.payload,
      );

      // Assert: the rebuild reproduces the worker's exact network channels
      // and weights (byte-equal to the mirrored spawn mind), and the second
      // rebuild is identical to the first.
      expect({
        restoredCarIndex: firstRebuilt.carIndex,
        rebuiltInputCount: firstRebuilt.network.input,
      }).toEqual({
        restoredCarIndex: RED_TRANSPORT_TEST_RIVAL_CAR_INDEX,
        rebuiltInputCount: RED_TRANSPORT_LEVEL_CHANNEL_COUNT,
      });
      expect(serializeNetworkWeights(secondRebuilt.network)).toEqual(
        serializeNetworkWeights(firstRebuilt.network),
      );
      expect(serializeNetworkWeights(firstRebuilt.network)).toEqual(
        RED_TRANSPORT_MIRRORED_RIVAL_SPAWN_WEIGHTS,
      );
    });
  });

  describe('RED-P4-23 — the elected live-feed laws', () => {
    it('feeds the LIVE rebuilt network only for the elected winner’s car', async () => {
      // Arrange: the transport helper does not exist yet (the gate names
      // it); an elected candidate with NO network (the current host feed
      // shape) and a resolution for the SAME car.
      const transport = await gateRedPanelNetworkTransport();
      const electedCandidate: NeatRacingNetworkPanelCandidate = {
        agentId: RED_TRANSPORT_TEST_RIVAL_CAR_INDEX,
        progress: 40,
        network: undefined,
      };
      const liveNetwork = new Network(
        RED_TRANSPORT_LEVEL_CHANNEL_COUNT,
        NEATRACING_NGE_CONTROLLER_OUTPUT_COUNT,
      );

      // Act: the feed resolution for the elected candidate's own car.
      const feed = transport.resolveNeatRacingPanelFeedNetwork(
        electedCandidate,
        { carIndex: RED_TRANSPORT_TEST_RIVAL_CAR_INDEX, network: liveNetwork },
      );

      // Assert: the candidate carries the SAME network object (the live
      // draw, not a copy, not a rebuild).
      expect(feed).toBe(liveNetwork);
    });

    it('keeps the idle path for a transport-pending winner — never a networked fallback', async () => {
      // Arrange: the transport helper (gate) + an elected candidate with no
      // network and a transport resolution still PENDING (undefined).
      const transport = await gateRedPanelNetworkTransport();
      const electedCandidate: NeatRacingNetworkPanelCandidate = {
        agentId: RED_TRANSPORT_TEST_RIVAL_CAR_INDEX,
        progress: 40,
        network: undefined,
      };

      // Act: the feed resolution with the pending transport.
      const feed = transport.resolveNeatRacingPanelFeedNetwork(
        electedCandidate,
        undefined,
      );

      // Assert: idle stays idle (undefined — the panel keeps its idle
      // cached frame; no fallback network is invented).
      expect(feed).toBeUndefined();
    });

    it('never feeds a networked runner-up: a foreign-car reply resolves idle for the elected candidate', async () => {
      // Arrange: the transport helper (gate) + an elected candidate whose
      // agentId does NOT match the resolution's car index.
      const transport = await gateRedPanelNetworkTransport();
      const runnerUpNetwork = new Network(
        RED_TRANSPORT_LEVEL_CHANNEL_COUNT,
        NEATRACING_NGE_CONTROLLER_OUTPUT_COUNT,
      );
      const electedCandidate: NeatRacingNetworkPanelCandidate = {
        agentId: RED_TRANSPORT_TEST_RIVAL_CAR_INDEX,
        progress: 40,
        network: undefined,
      };

      // Act: the same resolution handed to the elected candidate (foreign
      // car index) and to the matching candidate (the positive control).
      const foreignFeed = transport.resolveNeatRacingPanelFeedNetwork(
        electedCandidate,
        {
          carIndex: RED_TRANSPORT_TEST_RIVAL_CAR_INDEX + 1,
          network: runnerUpNetwork,
        },
      );
      const matchingFeed = transport.resolveNeatRacingPanelFeedNetwork(
        electedCandidate,
        {
          carIndex: RED_TRANSPORT_TEST_RIVAL_CAR_INDEX,
          network: runnerUpNetwork,
        },
      );

      // Assert: the carrier-match law — the foreign resolution stays idle
      // while the exact same resolution feeds the matching candidate.
      expect(foreignFeed).toBeUndefined();
      expect(matchingFeed).toBe(runnerUpNetwork);
    });
  });
});

describe('the panel network transport green companions (P4-S2C-transport)', () => {
  it('answers a not-yet-initialized session with the error reply and never an escape (the M1 reply law)', () => {
    // Arrange: a FRESH session that never booted (the same factory the
    // home's boot helper uses, WITHOUT the init request).
    if (!redTransportSimWorkerModule) {
      throw new Error('simulation-worker module not loaded in beforeAll');
    }
    const session =
      redTransportSimWorkerModule.NEATRACING_SIMULATION_WORKER_INTERNALS.createSimWorkerSession();

    // Act: the transport request against the un-booted session (the guard
    // REPLIES; it never throws).
    const responses = session.handleRequest(
      redTransportRequestFor(RED_TRANSPORT_TEST_RIVAL_CAR_INDEX),
    );
    const [reply] = responses as unknown as [RedPanelNetworkReplyEnvelope];
    const message = reply?.payload?.message;

    // Assert: exactly ONE error reply carrying a non-empty message (the
    // M1 reply law).
    expect({
      replyCount: responses.length,
      replyType: reply?.type,
      messageIsNonEmpty: typeof message === 'string' && message.length > 0,
    }).toEqual({
      replyCount: 1,
      replyType: 'error',
      messageIsNonEmpty: true,
    });
  });

  it('re-encodes the SAME ledger mind byte-identically on every request as a FRESH copy (the observational COPY law)', () => {
    // Arrange: the booted level-2 rival ladder + two identical requests.
    const session = createRedTransportRivalLadderSession();
    const firstReply = sendRedTransportRequest(
      session,
      RED_TRANSPORT_TEST_RIVAL_CAR_INDEX,
    );
    const secondReply = sendRedTransportRequest(
      session,
      RED_TRANSPORT_TEST_RIVAL_CAR_INDEX,
    );

    // Assert: the hex strings are byte-equal (the ledger mind never
    // changed) while the two payload objects are DISTINCT (the reply is a
    // fresh observational copy per request).
    expect(firstReply.payload.weights).toBe(secondReply.payload.weights);
    expect(firstReply.payload).not.toBe(secondReply.payload);
  });

  it('rejects a corrupted reply envelope with the NAMED RangeError (the restore guard law)', async () => {
    // Arrange: the transport module (the existing widened gate) + a real
    // reply whose envelope is corrupted only in its kind discriminator.
    const transport = await gateRedPanelNetworkTransport();
    const session = createRedTransportRivalLadderSession();
    const reply = sendRedTransportRequest(
      session,
      RED_TRANSPORT_TEST_RIVAL_CAR_INDEX,
    );
    const validEnvelope = {
      kind: RED_TRANSPORT_REPLY_ENVELOPE_KIND,
      carIndex: RED_TRANSPORT_TEST_RIVAL_CAR_INDEX,
      channelCount: RED_TRANSPORT_LEVEL_CHANNEL_COUNT,
      outputCount: NEATRACING_NGE_CONTROLLER_OUTPUT_COUNT,
      weights: reply.payload.weights ?? '',
    };

    // Act + Assert: the corrupted envelope is a loud NAMED rejection (the
    // RangeError guard law), never a silent fallback.
    expect(() =>
      transport.restoreNeatRacingPanelNetworkFromReply({
        ...validEnvelope,
        kind: 'not-panel-network',
      }),
    ).toThrow(RangeError);
  });

  it('resolves the feed as the resolution network on a carrier match and the elected network otherwise (the feed law companion)', async () => {
    // Arrange: the transport module (the existing widened gate) + the
    // mirror network (the home's existing mirror helper fills a network's
    // positional weights) + a foreign network for the negative case.
    const transport = await gateRedPanelNetworkTransport();
    const mirroredWeights = mirrorSpawnMindWeights(
      RED_TRANSPORT_TEST_SEED,
      RED_TRANSPORT_TEST_LADDER_LEVEL,
      RED_TRANSPORT_TEST_RIVAL_CAR_INDEX,
    );
    const mirrorNetwork = new Network(
      RED_TRANSPORT_LEVEL_CHANNEL_COUNT,
      NEATRACING_NGE_CONTROLLER_OUTPUT_COUNT,
    );
    for (let index = 0; index < mirrorNetwork.connections.length; index += 1) {
      mirrorNetwork.connections[index].weight = mirroredWeights[index];
    }
    const foreignNetwork = new Network(
      RED_TRANSPORT_LEVEL_CHANNEL_COUNT,
      NEATRACING_NGE_CONTROLLER_OUTPUT_COUNT,
    );
    const electedCandidate: NeatRacingNetworkPanelCandidate = {
      agentId: RED_TRANSPORT_TEST_RIVAL_CAR_INDEX,
      progress: 0,
      network: undefined,
    };
    const ownedCandidate: NeatRacingNetworkPanelCandidate = {
      agentId: RED_TRANSPORT_TEST_RIVAL_CAR_INDEX,
      progress: 0,
      network: mirrorNetwork,
    };

    // Act: the carrier match (the resolution names the candidate's own car)
    // and the foreign-car resolution.
    const carrierFeed = transport.resolveNeatRacingPanelFeedNetwork(
      electedCandidate,
      {
        carIndex: RED_TRANSPORT_TEST_RIVAL_CAR_INDEX,
        network: mirrorNetwork,
      },
    );
    const foreignFeed = transport.resolveNeatRacingPanelFeedNetwork(
      ownedCandidate,
      {
        carIndex: RED_TRANSPORT_TEST_RIVAL_CAR_INDEX + 1,
        network: foreignNetwork,
      },
    );

    // Assert: the carrier match hands the resolution's network; a foreign
    // resolution leaves the elected candidate's OWN network in place.
    expect(carrierFeed).toBe(mirrorNetwork);
    expect(foreignFeed).toBe(mirrorNetwork);
    expect(foreignFeed).not.toBe(foreignNetwork);
  });
});

describe('the panel network transport restore-guard companions (P4-S2C-transport, the review F5 totality pin)', () => {
  it('rejects every malformed reply envelope shape with the NAMED RangeError (the restore guard law)', async () => {
    // Arrange: the transport module (the existing widened gate) + a real
    // reply whose VALID envelope is copied verbatim into the corruption
    // table (the envelope law: kind/carIndex/channelCount/outputCount/
    // weights all come from the live reply).
    const transport = await gateRedPanelNetworkTransport();
    const session = createRedTransportRivalLadderSession();
    const reply = sendRedTransportRequest(
      session,
      RED_TRANSPORT_TEST_RIVAL_CAR_INDEX,
    );
    const validEnvelope = {
      kind: RED_TRANSPORT_REPLY_ENVELOPE_KIND,
      carIndex: RED_TRANSPORT_TEST_RIVAL_CAR_INDEX,
      channelCount: RED_TRANSPORT_LEVEL_CHANNEL_COUNT,
      outputCount: NEATRACING_NGE_CONTROLLER_OUTPUT_COUNT,
      weights: reply.payload.weights ?? '',
    };

    // The corruption table (the totality pin): every malformed envelope
    // shape the restore guard must reject, each named for the failure
    // report.
    const corruptions: ReadonlyArray<{
      readonly label: string;
      readonly envelope: unknown;
    }> = [
      {
        label: 'a null envelope (not an object at all)',
        envelope: null,
      },
      {
        label: 'car index 0 (the human car is never a rival slot)',
        envelope: { ...validEnvelope, carIndex: 0 },
      },
      {
        label: 'a zero channel count',
        envelope: { ...validEnvelope, channelCount: 0 },
      },
      {
        label: 'a zero output count',
        envelope: { ...validEnvelope, outputCount: 0 },
      },
      {
        label: 'non-hex weights',
        envelope: { ...validEnvelope, weights: 'ZZZZ' },
      },
      {
        label: 'one float32 of weights (the connection-count mismatch)',
        envelope: { ...validEnvelope, weights: '00000000' },
      },
    ];

    // Act + Assert: the guard names the rejection (the RangeError law) —
    // one row, one expectation.
    const captureRejectionName = (envelope: unknown): string => {
      try {
        transport.restoreNeatRacingPanelNetworkFromReply(envelope);
      } catch (error) {
        return error instanceof RangeError ? 'RangeError' : 'other-error';
      }
      return 'no-error';
    };
    for (const corruption of corruptions) {
      expect({
        label: corruption.label,
        thrown: captureRejectionName(corruption.envelope),
      }).toEqual({
        label: corruption.label,
        thrown: 'RangeError',
      });
    }
  });
});
