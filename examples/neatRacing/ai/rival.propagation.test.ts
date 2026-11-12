/**
 * Green tests for the NeatRacing rival propagation (`rival.propagation.ts` —
 * Step 04, slice P2-S3-D, TDD lane green-only).
 *
 * Contracts pinned here (AC-032 / spec section 9.6, E4):
 * - THE OPERATOR IS A WEIGHT-NUDGE PLANE ON SERIALIZED WEIGHTS: when a
 *   rival crosses the finish line, the OTHER rivals' minds are replaced by
 *   bounded seeded weight-nudge mutations of the crossing rival's
 *   serialized weights — ONE bounded MOD_WEIGHT round over the SMALL named
 *   delta range
 *   `[NEATRACING_RIVAL_SWAP_WEIGHT_DELTA_MIN, NEATRACING_RIVAL_SWAP_WEIGHT_DELTA_MAX]`
 *   (the src default plus/minus 1 is deliberately NOT reused: a rival
 *   swap is a NUDGE, not a full re-roll).
 * - THE RECYCLED `createCarGenome` PROTOCOL: the mind is the
 *   `CarGenome.serialize()` Float32Array of connection weights and the
 *   mutation mirrors the recycled MOD_WEIGHT semantics (per weight, one
 *   uniform delta sample, `min + u * (max - min)`), drawn in the PINNED
 *   draw order — targets ascending by car index, then weight index
 *   ascending — from the event's seeded xorshift32 stream (never
 *   `Math.random`).
 * - LVL-2 IS A DOCUMENTED NO-OP: a lone rival has nobody to propagate to
 *   — zero replacement minds and an empty propagated list in the payload.
 * - LVL-3 PROPAGATES TO THE THREE OTHER RIVALS (the four-rival case).
 * - LAP/LINE-BOUNDARY SWAPS ONLY: a mid-lap call (or an absent boundary
 *   flag) is a loud rejection, never a silent swap.
 * - THE MAIN AGENT IS STRUCTURALLY OUT OF REACH (the Blocked stop
 *   condition is a GUARD, not a promise): car 0 is the main agent's slot
 *   at EVERY level team layout (AC-033), so a rival set carrying car 0, a
 *   fractional index or a duplicate index is rejected.
 * - THE SWAP PAYLOAD IS THE SECTION 9.7 REPLAY ANCHOR: the crossing-network
 *   snapshot (carIndex + byte-exact hex weights) and the event's mutation
 *   seed ride into the reproducibility tuple's `opponentSnapshot` at each
 *   swap; the payload is JSON-SAFE BY CONSTRUCTION (hex weights, plain
 *   numbers, arrays) and survives the ONE base tuple serial path.
 * - THE CANONICAL PAYLOAD SERIAL (the D4 law from `promotion.rules.ts`):
 *   byte-exact hex weights (never a `toJSON`), key-sorted canonical
 *   serial — key-order invariant and byte-stable across repeated
 *   serializations.
 * - REPLAY FROM THE PAYLOAD ALONE (Level 2, same-runtime): the serialized
 *   payload rebuilds the operator's own replacement minds, byte-identical,
 *   without the original options, the tuple seed or any runtime state.
 *
 * Slice P4-S2-rival-propagation additions (the LIVE lane): the operator
 * semantics above are re-pinned against PUBLISHED payloads through the REAL
 * sim-worker session (imported headless after the fake worker scope binds —
 * the established worker-test pattern): the worker materializes each
 * rival's spawn mind deterministically from the tuple seed, publishes
 * `rivalSwaps` on the frames replies exactly at lap/line boundary
 * crossings, omits the field entirely on untouched frames, and its
 * serialized ledger compounds through the published payloads (each later
 * crossing propagates from the mind a previous event replaced). The lvl-2
 * single-rival rung stays a tested no-op in the live lane and the main
 * agent (car 0) is never a crossing anchor.
 *
 * @module
 */

import {
  serializeNeatRacingReproducibilityTuple,
  createNeatRacingXorShift32Stream,
  deriveNeatRacingCarNetworkChildSeed,
  deriveNeatRacingRivalSwapChildSeed,
  NEATRACING_RIVAL_SWAP_SEED_STRIDE,
  type NeatRacingReproducibilityTuple,
} from '../workers/simulation-worker/reproducibility.utils';
import {
  NEATRACING_TEAM_LAYOUTS,
  resolveNeatRacingObservationChannelCount,
} from './observation.assembler';
import { NEATRACING_NGE_CONTROLLER_OUTPUT_COUNT } from './nge.controller';
import { stepAutopilotInput } from '../physics/ai.autopilot';
import type { PhysicsCarState } from '../physics/physics.step.service';
import { generateTrack } from '../../racing_curriculum/track/track.generator';
import {
  NEATRACING_FIXED_TIMESTEP_MS,
  NEATRACING_MAX_CATCHUP_STEPS_PER_FRAME,
} from '../host/render-loop.utils';
import Network from '../../../src/architecture/network';
import type { NeatRacingFleetControlSeam } from '../workers/simulation-worker/nge.fleet.control';
import type {
  NeatRacingSimWorkerFramesPayload,
  NeatRacingSimWorkerSession,
} from '../workers/simulation-worker/simulation-worker';
import {
  NEATRACING_RIVAL_SWAP_PAYLOAD_KIND,
  NEATRACING_RIVAL_SWAP_WEIGHT_DELTA_MIN,
  NEATRACING_RIVAL_SWAP_WEIGHT_DELTA_MAX,
  propagateNeatRacingRivalWinnerWeights,
  restoreNeatRacingRivalSwapReplacementMinds,
  serializeNeatRacingRivalSwapPayload,
  toNeatRacingRivalSwapPayload,
  type NeatRacingRivalPropagationOptions,
  type NeatRacingRivalSwapPayload,
  type NeatRacingRivalSwapSnapshot,
  type NeatRacingRivalSwapReplacementMind,
} from './rival.propagation';

// ── Fixtures ─────────────────────────────────────────────────────────────────

/** The AC-026 tuple seed the fixtures replay. */
const TUPLE_SEED = 123_456_789;

/** The first swap-event index (the first stride slot of the swap stream). */
const EVENT_FIRST = 0;

/** The second swap-event index (the next stride slot). */
const EVENT_SECOND = 1;

/** The second level of the ladder (TP-12) — the lone-rival no-op rung. */
const LEVEL_SECOND = 2;

/** The third level of the ladder (TP-12) — the four-rival rung. */
const LEVEL_THIRD = 3;

/** The crossing rival's serialized mind fixture (four connection weights). */
const WINNER_WEIGHTS = new Float32Array([1, -0.5, 0.25, 2]);

/**
 * The byte-exact lowercase hex encoding of `WINNER_WEIGHTS`' raw float32
 * bytes (1 -> 0x3f800000, -0.5 -> 0xbf800000, 0.25 -> 0x3e800000,
 * 2 -> 0x40000000) in the D4 encoder's NATIVE byte order — the same order
 * the `promotion.rules.ts` D4 encoder emits (its own fixture pins
 * `Float32Array([1])` as `0000803f`). The bytes themselves run
 * `00 00 80 3f | 00 00 00 bf | 00 00 80 3e | 00 00 00 40`.
 */
const WINNER_WEIGHTS_HEX = '0000803f000000bf0000803e00000040';

/** The first swap event's derived mutation seed (the section 9.7.2 row). */
const FIRST_SWAP_MUTATION_SEED = deriveNeatRacingRivalSwapChildSeed(
  TUPLE_SEED,
  EVENT_FIRST,
);

// ── Fixture helpers ──────────────────────────────────────────────────────────

/**
 * Derives the session's rival car indices from the AUTHORITY (the E2 team
 * layouts): car 0 is the main agent's slot at every level (AC-033), so the
 * rivals are the slots whose team id is not team 0.
 *
 * @param level - The ladder level (1, 2 or 3).
 * @returns The level's rival car indices, ascending.
 */
function rivalCarIndicesForLevel(level: number): number[] {
  const layout =
    NEATRACING_TEAM_LAYOUTS[level as keyof typeof NEATRACING_TEAM_LAYOUTS];
  return layout
    .map((teamId, carIndex) => ({ carIndex, teamId }))
    .filter((slot) => slot.teamId !== 0)
    .map((slot) => slot.carIndex);
}

/**
 * Builds one lap/line-boundary swap call (the lvl-3 four-rival default).
 *
 * @param overrides - The fields to override.
 * @returns The swap call options.
 */
function buildSwapOptions(
  overrides: Partial<NeatRacingRivalPropagationOptions> = {},
): NeatRacingRivalPropagationOptions {
  return {
    rivalCarIndices: rivalCarIndicesForLevel(LEVEL_THIRD),
    crossingCarIndex: 2,
    winnerWeights: WINNER_WEIGHTS,
    tupleSeed: TUPLE_SEED,
    eventIndex: EVENT_FIRST,
    atLapLineBoundary: true,
    ...overrides,
  };
}

/**
 * Re-derives the expected replacement minds from the PUBLIC API only (the
 * seeded stream plus the named delta range), with `Math.fround` for the
 * float32 store parity the operator's Float32Array write provides.
 *
 * @param eventIndex - The swap-event index of the re-derived round.
 * @param targetCarIndices - The propagation targets (ascending).
 * @returns The expected replacement minds (ascending by car index).
 */
function deriveExpectedTargetWeights(
  eventIndex: number,
  targetCarIndices: readonly number[],
): NeatRacingRivalSwapReplacementMind[] {
  const stream = createNeatRacingXorShift32Stream(
    deriveNeatRacingRivalSwapChildSeed(TUPLE_SEED, eventIndex),
  );
  return [...targetCarIndices]
    .toSorted((first, second) => first - second)
    .map((carIndex) => {
      const weights = new Float32Array(WINNER_WEIGHTS.length);
      for (let index = 0; index < weights.length; index += 1) {
        const delta =
          NEATRACING_RIVAL_SWAP_WEIGHT_DELTA_MIN +
          stream() *
            (NEATRACING_RIVAL_SWAP_WEIGHT_DELTA_MAX -
              NEATRACING_RIVAL_SWAP_WEIGHT_DELTA_MIN);
        weights[index] = Math.fround(WINNER_WEIGHTS[index] + delta);
      }
      return { carIndex, weights };
    });
}

/**
 * Builds the base reproducibility tuple the swap payload is recorded into
 * (the AC-026 tuple's `opponentSnapshot` record).
 *
 * @param opponentSnapshot - The swap payload the tuple carries.
 * @returns The base tuple fixture.
 */
function buildBaseTupleFixture(
  opponentSnapshot: NeatRacingRivalSwapPayload,
): NeatRacingReproducibilityTuple {
  return {
    seed: TUPLE_SEED,
    agentCount: 5,
    packSchemaVersion: '1.0.0',
    opponentSnapshot,
    trackId: 'test-track',
    featureFlags: undefined,
  };
}

// ── The lvl-2 no-op ──────────────────────────────────────────────────────────

describe('the lvl-2 no-op (a lone rival has nobody to propagate to)', () => {
  it('returns zero replacement minds for the lone lvl-2 rival crossing', () => {
    const result = propagateNeatRacingRivalWinnerWeights(
      buildSwapOptions({
        rivalCarIndices: rivalCarIndicesForLevel(LEVEL_SECOND),
        crossingCarIndex: 1,
      }),
    );
    expect(result.replacements.length).toBe(0);
  });

  it('records an empty propagated list in the swap payload', () => {
    const { snapshot } = propagateNeatRacingRivalWinnerWeights(
      buildSwapOptions({
        rivalCarIndices: rivalCarIndicesForLevel(LEVEL_SECOND),
        crossingCarIndex: 1,
      }),
    );
    const payload = toNeatRacingRivalSwapPayload(snapshot);
    expect(payload.propagatedCarIndices).toEqual([]);
  });

  it('still anchors the crossing snapshot and the event seed', () => {
    const { snapshot } = propagateNeatRacingRivalWinnerWeights(
      buildSwapOptions({
        rivalCarIndices: rivalCarIndicesForLevel(LEVEL_SECOND),
        crossingCarIndex: 1,
      }),
    );
    const payload = toNeatRacingRivalSwapPayload(snapshot);
    expect(payload.crossing.carIndex).toBe(1);
    expect(payload.mutationSeed).toBe(FIRST_SWAP_MUTATION_SEED);
  });
});

// ── The lvl-3 propagation ────────────────────────────────────────────────────

describe('the lvl-3 propagation (four rivals — the winner re-seeds the others)', () => {
  it('replaces exactly the three non-crossing rivals in ascending order', () => {
    const result = propagateNeatRacingRivalWinnerWeights(buildSwapOptions());
    expect(
      result.replacements.map((replacement) => replacement.carIndex),
    ).toEqual([1, 3, 4]);
  });

  it('never hands the crossing rival a replacement mind', () => {
    const result = propagateNeatRacingRivalWinnerWeights(buildSwapOptions());
    expect(
      result.replacements.some((replacement) => replacement.carIndex === 2),
    ).toBe(false);
  });

  it("keeps every replacement on the winner's flat weight plane", () => {
    const result = propagateNeatRacingRivalWinnerWeights(buildSwapOptions());
    expect(
      result.replacements.every(
        (replacement) => replacement.weights.length === WINNER_WEIGHTS.length,
      ),
    ).toBe(true);
  });

  it('writes exactly the seeded nudge math in the pinned draw order', () => {
    const result = propagateNeatRacingRivalWinnerWeights(buildSwapOptions());
    expect(result.replacements).toEqual(
      deriveExpectedTargetWeights(EVENT_FIRST, [1, 3, 4]),
    );
  });

  it("never mutates the winner's serialized weights", () => {
    const winnerWeights = new Float32Array([1, -0.5, 0.25, 2]);
    propagateNeatRacingRivalWinnerWeights(buildSwapOptions({ winnerWeights }));
    expect(winnerWeights).toEqual(new Float32Array([1, -0.5, 0.25, 2]));
  });
});

// ── The main-agent exclusion ─────────────────────────────────────────────────

describe('the main-agent exclusion (the Blocked stop condition is a guard)', () => {
  it("rejects a rival set that includes the main agent's car 0 slot", () => {
    expect(() =>
      propagateNeatRacingRivalWinnerWeights(
        buildSwapOptions({ rivalCarIndices: [0, 1, 2, 3, 4] }),
      ),
    ).toThrow(RangeError);
  });

  it('rejects a fractional rival car index', () => {
    expect(() =>
      propagateNeatRacingRivalWinnerWeights(
        buildSwapOptions({ rivalCarIndices: [1, 2.5, 3] }),
      ),
    ).toThrow(RangeError);
  });

  it('rejects duplicate rival car indices', () => {
    expect(() =>
      propagateNeatRacingRivalWinnerWeights(
        buildSwapOptions({ rivalCarIndices: [1, 1, 2, 3, 4] }),
      ),
    ).toThrow(RangeError);
  });
});

// ── The LAP/line boundary timing ────────────────────────────────────────────

describe('the LAP/line boundary timing (E4 — never mid-lap)', () => {
  it('rejects a mid-lap call', () => {
    expect(() =>
      propagateNeatRacingRivalWinnerWeights(
        buildSwapOptions({ atLapLineBoundary: false }),
      ),
    ).toThrow(RangeError);
  });

  it('rejects an absent boundary flag', () => {
    expect(() =>
      propagateNeatRacingRivalWinnerWeights(
        buildSwapOptions({
          atLapLineBoundary: undefined as unknown as boolean,
        }),
      ),
    ).toThrow(RangeError);
  });
});

// ── The crossing anchor and the recycled weight guards ───────────────────────

describe('the crossing anchor and the recycled weight guards', () => {
  it('rejects a crossing car index outside the rival set', () => {
    expect(() =>
      propagateNeatRacingRivalWinnerWeights(
        buildSwapOptions({ crossingCarIndex: 9 }),
      ),
    ).toThrow(RangeError);
  });

  it('rejects empty crossing weights', () => {
    expect(() =>
      propagateNeatRacingRivalWinnerWeights(
        buildSwapOptions({ winnerWeights: new Float32Array([]) }),
      ),
    ).toThrow(RangeError);
  });

  it('rejects a non-finite crossing weight', () => {
    expect(() =>
      propagateNeatRacingRivalWinnerWeights(
        buildSwapOptions({
          winnerWeights: new Float32Array([1, Number.NaN, 2]),
        }),
      ),
    ).toThrow(RangeError);
  });

  it('rejects a non-finite tuple seed through the shared derivation table', () => {
    expect(() =>
      propagateNeatRacingRivalWinnerWeights(
        buildSwapOptions({ tupleSeed: Number.NaN }),
      ),
    ).toThrow(RangeError);
  });

  it('rejects a negative event index through the shared derivation table', () => {
    expect(() =>
      propagateNeatRacingRivalWinnerWeights(
        buildSwapOptions({ eventIndex: -1 }),
      ),
    ).toThrow(RangeError);
  });
});

// ── The crossing-network snapshot ────────────────────────────────────────────

describe('the crossing-network snapshot (the runtime replay anchor)', () => {
  it('anchors the crossing rival and the event index', () => {
    const { snapshot } =
      propagateNeatRacingRivalWinnerWeights(buildSwapOptions());
    expect(snapshot.crossing.carIndex).toBe(2);
    expect(snapshot.eventIndex).toBe(EVENT_FIRST);
  });

  it('carries the derived mutation seed', () => {
    const { snapshot } =
      propagateNeatRacingRivalWinnerWeights(buildSwapOptions());
    expect(snapshot.mutationSeed).toBe(FIRST_SWAP_MUTATION_SEED);
  });

  it('copies the crossing weights so the snapshot owns its bytes', () => {
    const winnerWeights = new Float32Array(WINNER_WEIGHTS);
    const { snapshot } = propagateNeatRacingRivalWinnerWeights(
      buildSwapOptions({ winnerWeights }),
    );
    winnerWeights[0] = 9;
    expect(snapshot.crossing.weights[0]).toBe(1);
  });
});

// ── The swap payload ─────────────────────────────────────────────────────────

describe('the swap payload (the opponentSnapshot record — JSON-safe by construction)', () => {
  it('encodes the crossing weights as byte-exact lowercase hex', () => {
    const { snapshot } =
      propagateNeatRacingRivalWinnerWeights(buildSwapOptions());
    const payload = toNeatRacingRivalSwapPayload(snapshot);
    expect(payload.crossing.weights).toBe(WINNER_WEIGHTS_HEX);
  });

  it('carries the discriminator and the event fields', () => {
    const { snapshot } =
      propagateNeatRacingRivalWinnerWeights(buildSwapOptions());
    const payload = toNeatRacingRivalSwapPayload(snapshot);
    expect(payload.kind).toBe('rival-swap');
    expect(payload.eventIndex).toBe(EVENT_FIRST);
    expect(payload.mutationSeed).toBe(FIRST_SWAP_MUTATION_SEED);
  });

  it('survives the ONE base tuple serial path', () => {
    const { snapshot } =
      propagateNeatRacingRivalWinnerWeights(buildSwapOptions());
    const payload = toNeatRacingRivalSwapPayload(snapshot);
    const tuple = buildBaseTupleFixture(payload);
    const parsed = JSON.parse(
      serializeNeatRacingReproducibilityTuple(tuple),
    ) as { opponentSnapshot: NeatRacingRivalSwapPayload };
    expect(parsed.opponentSnapshot).toEqual(payload);
  });

  it('rejects a payload whose weights field carries a typed array', () => {
    const { snapshot } =
      propagateNeatRacingRivalWinnerWeights(buildSwapOptions());
    const payload = toNeatRacingRivalSwapPayload(snapshot);
    const tampered = {
      ...payload,
      crossing: {
        ...payload.crossing,
        weights: new Float32Array([1]) as unknown as string,
      },
    };
    expect(() => serializeNeatRacingRivalSwapPayload(tampered)).toThrow(
      RangeError,
    );
  });
});

// ── The replay path ──────────────────────────────────────────────────────────

describe('the replay path — swap events rebuild from the tuple alone', () => {
  it('rebuilds the replacements from the payload alone', () => {
    const { snapshot, replacements } =
      propagateNeatRacingRivalWinnerWeights(buildSwapOptions());
    const payload = toNeatRacingRivalSwapPayload(snapshot);
    expect(restoreNeatRacingRivalSwapReplacementMinds(payload)).toEqual(
      replacements,
    );
  });

  it('is replay-exact across repeated restores', () => {
    const { snapshot } =
      propagateNeatRacingRivalWinnerWeights(buildSwapOptions());
    const payload = toNeatRacingRivalSwapPayload(snapshot);
    const firstRestore = restoreNeatRacingRivalSwapReplacementMinds(payload);
    const secondRestore = restoreNeatRacingRivalSwapReplacementMinds(payload);
    expect(secondRestore).toEqual(firstRestore);
  });

  it('advances the mutation stream per event by the recorded stride', () => {
    const secondEventSeed = deriveNeatRacingRivalSwapChildSeed(
      TUPLE_SEED,
      EVENT_SECOND,
    );
    expect(secondEventSeed).toBe(
      (FIRST_SWAP_MUTATION_SEED + NEATRACING_RIVAL_SWAP_SEED_STRIDE) >>> 0,
    );
  });

  it('different events nudge different minds', () => {
    const firstEvent =
      propagateNeatRacingRivalWinnerWeights(buildSwapOptions());
    const secondEvent = propagateNeatRacingRivalWinnerWeights(
      buildSwapOptions({ eventIndex: EVENT_SECOND }),
    );
    expect(secondEvent.replacements).not.toEqual(firstEvent.replacements);
  });

  it('round-trips through the canonical serial (serialize -> parse -> restore)', () => {
    const { snapshot, replacements } =
      propagateNeatRacingRivalWinnerWeights(buildSwapOptions());
    const payload = toNeatRacingRivalSwapPayload(snapshot);
    const parsed = JSON.parse(
      serializeNeatRacingRivalSwapPayload(payload),
    ) as NeatRacingRivalSwapPayload;
    expect(restoreNeatRacingRivalSwapReplacementMinds(parsed)).toEqual(
      replacements,
    );
  });
});

// ── The canonical payload serial ─────────────────────────────────────────────

describe('the canonical payload serial (byte-exact replay anchor)', () => {
  it('is byte-stable across repeated serializations', () => {
    const { snapshot } =
      propagateNeatRacingRivalWinnerWeights(buildSwapOptions());
    const payload = toNeatRacingRivalSwapPayload(snapshot);
    expect(serializeNeatRacingRivalSwapPayload(payload)).toBe(
      serializeNeatRacingRivalSwapPayload(payload),
    );
  });

  it('is key-order invariant', () => {
    const { snapshot } =
      propagateNeatRacingRivalWinnerWeights(buildSwapOptions());
    const payload = toNeatRacingRivalSwapPayload(snapshot);
    const handShuffled = {
      propagatedCarIndices: payload.propagatedCarIndices,
      mutationSeed: payload.mutationSeed,
      kind: payload.kind,
      eventIndex: payload.eventIndex,
      crossing: {
        weights: payload.crossing.weights,
        carIndex: payload.crossing.carIndex,
      },
    } as NeatRacingRivalSwapPayload;
    expect(serializeNeatRacingRivalSwapPayload(handShuffled)).toBe(
      serializeNeatRacingRivalSwapPayload(payload),
    );
  });

  it('embeds the canonical sorted shape with the hex weights', () => {
    const { snapshot } =
      propagateNeatRacingRivalWinnerWeights(buildSwapOptions());
    const payload = toNeatRacingRivalSwapPayload(snapshot);
    const serial = serializeNeatRacingRivalSwapPayload(payload);
    expect(serial).toContain('"kind":"rival-swap"');
    expect(serial).toContain(`"weights":"${WINNER_WEIGHTS_HEX}"`);
    expect(serial).toContain('"eventIndex":0');
  });

  it('rejects a NaN mutation seed', () => {
    const { snapshot } =
      propagateNeatRacingRivalWinnerWeights(buildSwapOptions());
    const payload = toNeatRacingRivalSwapPayload(snapshot);
    const tampered = { ...payload, mutationSeed: Number.NaN };
    expect(() => serializeNeatRacingRivalSwapPayload(tampered)).toThrow(
      RangeError,
    );
  });

  it('rejects an uppercase-hex weights field', () => {
    const { snapshot } =
      propagateNeatRacingRivalWinnerWeights(buildSwapOptions());
    const payload = toNeatRacingRivalSwapPayload(snapshot);
    const tampered = {
      ...payload,
      crossing: {
        ...payload.crossing,
        weights: '3F800000BF8000003E80000040000000',
      },
    };
    expect(() => serializeNeatRacingRivalSwapPayload(tampered)).toThrow(
      RangeError,
    );
  });

  it('rejects non-ascending propagated indices', () => {
    const { snapshot } =
      propagateNeatRacingRivalWinnerWeights(buildSwapOptions());
    const payload = toNeatRacingRivalSwapPayload(snapshot);
    const tampered = { ...payload, propagatedCarIndices: [3, 1, 4] };
    expect(() => serializeNeatRacingRivalSwapPayload(tampered)).toThrow(
      RangeError,
    );
  });
});

// ── The runtime snapshot guard (fix packet iteration 1, FIX 4) ───────────────

describe('the runtime snapshot guard (toNeatRacingRivalSwapPayload rejections)', () => {
  it('rejects a fractional or negative event index', () => {
    const { snapshot } =
      propagateNeatRacingRivalWinnerWeights(buildSwapOptions());
    expect(() =>
      toNeatRacingRivalSwapPayload({
        ...snapshot,
        eventIndex: -1,
      } as unknown as NeatRacingRivalSwapSnapshot),
    ).toThrow(RangeError);
    expect(() =>
      toNeatRacingRivalSwapPayload({
        ...snapshot,
        eventIndex: 1.5,
      } as unknown as NeatRacingRivalSwapSnapshot),
    ).toThrow(RangeError);
  });

  it('rejects a mutation seed outside the unsigned 32-bit seed range', () => {
    const { snapshot } =
      propagateNeatRacingRivalWinnerWeights(buildSwapOptions());
    expect(() =>
      toNeatRacingRivalSwapPayload({
        ...snapshot,
        mutationSeed: 0x100000000,
      } as unknown as NeatRacingRivalSwapSnapshot),
    ).toThrow(RangeError);
  });

  it('rejects a main-agent-slot crossing index', () => {
    const { snapshot } =
      propagateNeatRacingRivalWinnerWeights(buildSwapOptions());
    expect(() =>
      toNeatRacingRivalSwapPayload({
        ...snapshot,
        crossing: { ...snapshot.crossing, carIndex: 0 },
      } as unknown as NeatRacingRivalSwapSnapshot),
    ).toThrow(RangeError);
  });

  it('rejects empty or non-finite crossing weights', () => {
    const { snapshot } =
      propagateNeatRacingRivalWinnerWeights(buildSwapOptions());
    expect(() =>
      toNeatRacingRivalSwapPayload({
        ...snapshot,
        crossing: { ...snapshot.crossing, weights: new Float32Array([]) },
      } as unknown as NeatRacingRivalSwapSnapshot),
    ).toThrow(RangeError);
    expect(() =>
      toNeatRacingRivalSwapPayload({
        ...snapshot,
        crossing: {
          ...snapshot.crossing,
          weights: new Float32Array([1, Number.NaN]),
        },
      } as unknown as NeatRacingRivalSwapSnapshot),
    ).toThrow(RangeError);
  });

  it('rejects non-ascending propagated indices', () => {
    const { snapshot } =
      propagateNeatRacingRivalWinnerWeights(buildSwapOptions());
    expect(() =>
      toNeatRacingRivalSwapPayload({
        ...snapshot,
        propagatedCarIndices: [3, 1],
      } as unknown as NeatRacingRivalSwapSnapshot),
    ).toThrow(RangeError);
  });

  it('rejects propagated indices carrying the crossing rival', () => {
    const { snapshot } =
      propagateNeatRacingRivalWinnerWeights(buildSwapOptions());
    expect(() =>
      toNeatRacingRivalSwapPayload({
        ...snapshot,
        propagatedCarIndices: [2, 3],
      } as unknown as NeatRacingRivalSwapSnapshot),
    ).toThrow(RangeError);
  });

  it('rejects a snapshot record missing its crossing snapshot', () => {
    const { snapshot } =
      propagateNeatRacingRivalWinnerWeights(buildSwapOptions());
    expect(() =>
      toNeatRacingRivalSwapPayload({
        ...snapshot,
        crossing: undefined,
      } as unknown as NeatRacingRivalSwapSnapshot),
    ).toThrow(RangeError);
  });

  it('rejects a payload record missing its crossing snapshot', () => {
    const { snapshot } =
      propagateNeatRacingRivalWinnerWeights(buildSwapOptions());
    const payload = toNeatRacingRivalSwapPayload(snapshot);
    expect(() =>
      serializeNeatRacingRivalSwapPayload({
        ...payload,
        crossing: undefined,
      } as unknown as NeatRacingRivalSwapPayload),
    ).toThrow(RangeError);
  });
});

// ── The bounded-nudge invariants (fix packet iteration 1, FIX 5) ─────────────

describe('the bounded-nudge invariants (the E4 small-delta law)', () => {
  it('nudges every weight within the bounded delta range', () => {
    const { replacements } =
      propagateNeatRacingRivalWinnerWeights(buildSwapOptions());
    const allWithinBounds = replacements.every((mind) =>
      mind.weights.every(
        (weight, index) =>
          Math.abs(weight - WINNER_WEIGHTS[index]) <=
          NEATRACING_RIVAL_SWAP_WEIGHT_DELTA_MAX,
      ),
    );
    expect(allWithinBounds).toBe(true);
  });

  it('stays symmetric and strictly narrower than the src MOD_WEIGHT default', () => {
    expect(NEATRACING_RIVAL_SWAP_WEIGHT_DELTA_MIN).toBe(
      -NEATRACING_RIVAL_SWAP_WEIGHT_DELTA_MAX,
    );
    expect(NEATRACING_RIVAL_SWAP_WEIGHT_DELTA_MAX).toBeLessThanOrEqual(1);
  });

  it('rejects a NaN-decoding hex payload on both public payload entries', () => {
    const { snapshot } =
      propagateNeatRacingRivalWinnerWeights(buildSwapOptions());
    const payload = toNeatRacingRivalSwapPayload(snapshot);
    const nanHexPayload = {
      ...payload,
      crossing: { ...payload.crossing, weights: '0000c07f' },
    } as unknown as NeatRacingRivalSwapPayload;
    expect(() => serializeNeatRacingRivalSwapPayload(nanHexPayload)).toThrow(
      RangeError,
    );
    expect(() =>
      restoreNeatRacingRivalSwapReplacementMinds(nanHexPayload),
    ).toThrow(RangeError);
  });
});

// ── The LIVE lane (slice P4-S2-rival-propagation, AC-032) ────────────────────
//
// The describes above pin the OPERATOR's semantics in isolation; this section
// pins the LIVE worker-plane integration the served `ai` mode drives: a REAL
// sim-worker session (imported headless after the fake worker scope binds —
// the established worker-test pattern; the node test environment has no
// `self` and the worker module boots its inbox adapter at import time) runs
// the served track geometry with the REAL autopilot driving every rival, and
// the frames replies must publish the rival-swap events exactly the way the
// serialized-plane operator defines them:
//
// - the worker materializes each rival's SPAWN mind from the tuple seed (the
//   same deterministic reconstruction the adaptation plane uses: a network
//   of the level's observation channel count and the controller output
//   count, seeded from the per-car child seed), serialized positionally;
// - a rival crossing the lap/line boundary resolves ONE swap event whose
//   numbering, event seed, draw order and replacement minds follow the
//   operator exactly;
// - the published payload IS the section 9.7 replay anchor (JSON-safe hex
//   weights, plain numbers) and the worker's serialized ledger COMPOUNDS
//   through it: each later crossing propagates from the mind a previous
//   event replaced;
// - the lvl-2 single-rival rung stays a tested no-op in the live lane and
//   the main agent (car 0) is never a crossing anchor;
// - frames with NO rival crossing omit the `rivalSwaps` field entirely (the
//   byte-shape mandate: untouched frames keep their exact prior shape).
//
// Educational note — the drive pacing: this lane runs the REAL render-loop
// pacing (the RED home's rival drive MOCKS `advanceFrameSteps` to release a
// flat 8 ticks per frame, this lane does not). The real `advanceFrameSteps`
// caps catch-up at `NEATRACING_MAX_CATCHUP_STEPS_PER_FRAME` ticks per drawn
// frame, so the drive steps exactly one ceiling-full of fixed ticks per
// paced frame: a frame step of four 60 Hz timesteps releases exactly four
// ticks, deterministically, with no accumulator drift.

/** Live-lane drive pace: the real render-loop catch-up ceiling per frame. */
const LIVE_LANE_TICKS_PER_FRAME = NEATRACING_MAX_CATCHUP_STEPS_PER_FRAME;

/**
 * Live-lane frame step in ms: one ceiling-full of fixed 60 Hz timesteps per
 * paced frame (deterministically releases exactly four ticks each frame).
 */
const LIVE_LANE_FRAME_STEP_MS =
  NEATRACING_FIXED_TIMESTEP_MS * LIVE_LANE_TICKS_PER_FRAME;

/**
 * Live-lane drive budget in frames: comfortably more than one autopilot lap
 * of the served track (the served 'medium' seed-42 geometry closes a lap in
 * roughly 19 s — about 1,150 fixed ticks — so 900 paced frames of four
 * ticks each ≈ three full laps, leaving a wide margin while keeping the
 * drive bounded).
 */
const LIVE_LANE_FRAME_BUDGET = 900;

/** The served-track fixture seed (the autopilot suite's driven geometry). */
const LIVE_LANE_TRACK_SEED = 42;

/** The served layout version of the live-lane fixture. */
const LIVE_LANE_LAYOUT_VERSION = 1;

/**
 * The served size bucket of the live-lane fixture (the served demo boot's
 * own `NEATRACING_DEMO_TRACK_GENERATION` law: seed 42, layout version 1,
 * size bucket 'large' — `examples/neatRacing/browser-entry/browser-entry.ts`).
 */
const LIVE_LANE_SIZE_BUCKET = 'large';

/** The live-lane fleet size at the lvl-3 rung (car 0 plus four rivals). */
const LIVE_LANE_AGENT_COUNT_LVL_THIRD = 5;

/** The live-lane fleet size at the lvl-2 rung (car 0 plus one rival). */
const LIVE_LANE_AGENT_COUNT_LVL_SECOND = 2;

/** The fake worker scope the sim-worker module binds its inbox adapter onto. */
const liveLaneSimWorkerScope: {
  onmessage: unknown;
  postMessage: (message: unknown) => void;
} = {
  onmessage: undefined,
  postMessage: (): void => {},
};

/** The REAL `self` property descriptor, captured before the fake binds. */
const liveLaneOriginalSelfDescriptor = Object.getOwnPropertyDescriptor(
  globalThis,
  'self',
);

// The node test environment has no `self` and the sim-worker module boots its
// inbox adapter (`self.onmessage`) at import time, so the fake scope binds at
// module scope — BEFORE the dynamic import below can ever run.
Object.defineProperty(globalThis, 'self', {
  configurable: true,
  value: liveLaneSimWorkerScope,
});

afterAll(() => {
  if (liveLaneOriginalSelfDescriptor === undefined) {
    Object.defineProperty(globalThis, 'self', {
      configurable: true,
      value: undefined,
    });
  } else {
    Object.defineProperty(globalThis, 'self', liveLaneOriginalSelfDescriptor);
  }
});

/**
 * The sim-worker module, dynamically imported AFTER the fake scope binds
 * (the import boots the inbox adapter, so it must never be a static import).
 */
let liveLaneSimWorkerModule: typeof import('../workers/simulation-worker/simulation-worker');

beforeAll(async () => {
  liveLaneSimWorkerModule =
    await import('../workers/simulation-worker/simulation-worker');
});

// The served-track fixture: the exact generated geometry the autopilot suite
// drives (a fully closed, self-validating loop), shared by every live-lane
// boot below.
const liveLaneTrackSpec = generateTrack({
  seed: LIVE_LANE_TRACK_SEED,
  layoutVersion: LIVE_LANE_LAYOUT_VERSION,
  sizeBucket: LIVE_LANE_SIZE_BUCKET,
});

/**
 * The live-lane controller double: the REAL autopilot drives every car it is
 * handed (the same `stepAutopilotInput` seam the RED home's rival drive uses
 * — and the worker's fleet-control seam clamps and derives the turn rate).
 *
 * Car 0 stays on the human lane (the worker samples D5 input for it only in
 * `manual` mode and boots `ai` by default), so car 0 is parked at its spawn
 * pose and can never become a crossing anchor: the live lane inherits the
 * main-agent independence for free.
 */
function buildLiveLaneControllerDouble(): NeatRacingFleetControlSeam {
  return {
    computeControl(fleet: readonly PhysicsCarState[], carIndex: number) {
      return stepAutopilotInput(
        fleet[carIndex],
        liveLaneTrackSpec.splineSamples,
      );
    },
  };
}

/**
 * Boots one live-lane session from the served fixture (the init reply is
 * asserted to be the single `ready` acknowledgement).
 *
 * @param level - The served opponent-snapshot level (the rival rung).
 * @param agentCount - The fleet size the session boots with.
 * @returns The booted sim-worker session.
 */
function bootLiveLaneSession(
  level: 1 | 2 | 3,
  agentCount: number,
): NeatRacingSimWorkerSession {
  const session = liveLaneSimWorkerModule.createSimWorkerSession();
  const responses = session.handleRequest({
    type: 'init',
    payload: {
      seed: LIVE_LANE_TRACK_SEED,
      agentCount,
      packSchemaVersion: '1.0.0',
      opponentSnapshot: { level, carCount: agentCount },
      trackId: 'fixture-live-lane-served-track',
      featureFlags: undefined,
      trackSpec: liveLaneTrackSpec,
      raceMode: 'ai',
      controller: buildLiveLaneControllerDouble(),
    },
  });
  if (responses.length !== 1 || responses[0].type !== 'ready') {
    throw new Error('live lane: the served boot did not acknowledge ready');
  }
  return session;
}

/**
 * Drives one live-lane session frame by frame, harvesting every frames
 * reply. The FIRST paced frame (t = 0) is the render-loop anchor (delta 0,
 * zero ticks — exactly like the real loop's first drawn frame), then every
 * paced frame releases exactly one ceiling-full of fixed ticks.
 *
 * @param session - The booted sim-worker session.
 * @param stop - Called with the swap-event total so far after each frame;
 *   when it returns true the drive stops (the budget never runs past).
 * @returns Every frames payload observed during the drive, in order.
 */
function driveLiveLaneFrames(
  session: NeatRacingSimWorkerSession,
  stop: (swapsSoFar: number) => boolean,
): NeatRacingSimWorkerFramesPayload[] {
  const payloads: NeatRacingSimWorkerFramesPayload[] = [];
  let swapsSoFar = 0;
  for (
    let frameIndex = 0;
    frameIndex < LIVE_LANE_FRAME_BUDGET;
    frameIndex += 1
  ) {
    const responses = session.handleRequest({
      type: 'frame',
      payload: { timestampMs: frameIndex * LIVE_LANE_FRAME_STEP_MS },
    });
    if (responses.length !== 1 || responses[0].type !== 'frames') {
      throw new Error(
        `live lane: unexpected reply at frame ${frameIndex} (${responses.length} replies)`,
      );
    }
    const payload = (
      responses[0] as { payload: NeatRacingSimWorkerFramesPayload }
    ).payload;
    payloads.push(payload);
    swapsSoFar += payload.rivalSwaps?.length ?? 0;
    if (stop(swapsSoFar)) {
      break;
    }
  }
  return payloads;
}

/** Collects every published rival-swap event across a live drive. */
function collectLiveLaneSwaps(
  payloads: readonly NeatRacingSimWorkerFramesPayload[],
): NeatRacingRivalSwapPayload[] {
  return payloads.flatMap((payload) => payload.rivalSwaps ?? []);
}

/** Serializes one network positionally: connection i's weight at slot i. */
function serializeLiveLaneNetworkWeights(network: Network): Float32Array {
  const weights = new Float32Array(network.connections.length);
  for (let index = 0; index < network.connections.length; index += 1) {
    weights[index] = network.connections[index].weight;
  }
  return weights;
}

/**
 * Materializes one rival's spawn mind exactly the way the worker's serialized
 * ledger does: a fresh network of the level's observation channel count and
 * the controller output count, seeded from the per-car child seed of the
 * tuple seed, serialized positionally (the CarGenome.serialize() plane).
 */
function materializeLiveLaneSpawnMind(
  carIndex: number,
  level: 1 | 2 | 3,
  seed: number,
): Float32Array {
  const network = new Network(
    resolveNeatRacingObservationChannelCount(level),
    NEATRACING_NGE_CONTROLLER_OUTPUT_COUNT,
    { seed: deriveNeatRacingCarNetworkChildSeed(seed, carIndex) },
  );
  return serializeLiveLaneNetworkWeights(network);
}

/** Re-encodes serialized weights as the payload's byte-exact lowercase hex. */
function hexOfSerializedWeights(weights: Float32Array): string {
  return Array.from(
    new Uint8Array(weights.buffer, weights.byteOffset, weights.byteLength),
    (byte) => byte.toString(16).padStart(2, '0'),
  ).join('');
}

/** Decodes the payload's byte-exact lowercase hex weights (native byte order). */
function decodeLiveLaneHexWeights(hex: string): Float32Array {
  const byteLength = hex.length / 2;
  if (byteLength % 4 !== 0) {
    throw new Error('live lane: hex weights do not decode to float32 words');
  }
  const bytes = new Uint8Array(byteLength);
  for (let byteIndex = 0; byteIndex < byteLength; byteIndex += 1) {
    bytes[byteIndex] = parseInt(
      hex.slice(byteIndex * 2, byteIndex * 2 + 2),
      16,
    );
  }
  return new Float32Array(bytes.buffer);
}

/**
 * Asserts that the published swap events follow the operator EXACTLY: the
 * event numbering is contiguous from zero, every event seed is the section
 * 9.7.2 rival-swap row, every crossing anchor is a rival's serialized LEDGER
 * mind (its spawn mind until a previous event replaces it), every propagated
 * list matches the operator's own replacements, every payload survives the
 * canonical serial, and the worker's ledger compounds through the payloads.
 */
function assertLiveLaneSwapsFollowTheOperator(
  swaps: readonly NeatRacingRivalSwapPayload[],
  rivals: readonly number[],
  level: 1 | 2 | 3,
  seed: number,
): void {
  // The mirrored ledger: each rival's serialized mind, evolving exactly by
  // the operator's semantics (spawn materialization, then replacements).
  const ledger = new Map<number, Float32Array>();
  for (const rivalCarIndex of rivals) {
    ledger.set(
      rivalCarIndex,
      materializeLiveLaneSpawnMind(rivalCarIndex, level, seed),
    );
  }

  for (let swapIndex = 0; swapIndex < swaps.length; swapIndex += 1) {
    const payload = swaps[swapIndex];
    // The event numbering is contiguous from zero (the per-round counter).
    expect(payload.eventIndex).toBe(swapIndex);
    // The event seed is the rival-swap row of the tuple (seed stride 2).
    expect(payload.mutationSeed).toBe(
      deriveNeatRacingRivalSwapChildSeed(seed, payload.eventIndex),
    );
    // The payload kind and the JSON-safe publication law (RED-P4-6).
    expect(payload.kind).toBe(NEATRACING_RIVAL_SWAP_PAYLOAD_KIND);
    expect(serializeNeatRacingRivalSwapPayload(payload).length).toBeGreaterThan(
      0,
    );
    // The main agent is never a crossing anchor in the live lane.
    expect(payload.crossing.carIndex).toBeGreaterThan(0);

    // The crossing weights ARE the rival's serialized ledger mind at the
    // moment it crossed (the compounding law).
    const expectedLedgerMind = ledger.get(payload.crossing.carIndex);
    if (expectedLedgerMind === undefined) {
      throw new Error(
        `live lane: crossing rival ${payload.crossing.carIndex} has no ledger mind`,
      );
    }
    const crossingWeights = decodeLiveLaneHexWeights(payload.crossing.weights);
    expect(hexOfSerializedWeights(crossingWeights)).toBe(
      hexOfSerializedWeights(expectedLedgerMind),
    );

    // The event resolves EXACTLY once through the operator over the
    // crossing's serialized weights: the same draw order and seeds, so the
    // published propagated list and mutation seed match the re-resolution.
    const { snapshot, replacements } = propagateNeatRacingRivalWinnerWeights({
      rivalCarIndices: [...rivals],
      crossingCarIndex: payload.crossing.carIndex,
      winnerWeights: crossingWeights,
      tupleSeed: seed,
      eventIndex: payload.eventIndex,
      atLapLineBoundary: true,
    });
    expect(payload.mutationSeed).toBe(snapshot.mutationSeed);
    expect([...payload.propagatedCarIndices]).toEqual([
      ...snapshot.propagatedCarIndices,
    ]);

    // The ledger compounding: this event's replacement minds are what the
    // targets' NEXT crossings carry.
    for (const replacement of replacements) {
      ledger.set(replacement.carIndex, replacement.weights);
    }
  }
}

describe('the LIVE lane — the worker-plane rival swap at the lap/line boundary (P4-S2, AC-032)', () => {
  it('publishes operator-exact rival-swap payloads when the real autopilot drives the lvl-3 four-rival round past the lap line', () => {
    const rivals = rivalCarIndicesForLevel(LEVEL_THIRD);
    const session = bootLiveLaneSession(
      LEVEL_THIRD,
      LIVE_LANE_AGENT_COUNT_LVL_THIRD,
    );

    // Act: drive the served round until the FOUR rivals have each crossed
    // once (a full event round) or the frame budget is exhausted.
    const payloads = driveLiveLaneFrames(
      session,
      (swapsSoFar) => swapsSoFar >= rivals.length,
    );
    const swaps = collectLiveLaneSwaps(payloads);

    // The drive is not VACUOUS: the real autopilot really closes the lap and
    // all four rivals really cross within the budget.
    expect(swaps.length).toBeGreaterThanOrEqual(rivals.length);

    // Untouched frames keep their exact prior shape: before the first
    // crossing the frames payload carries NO `rivalSwaps` field at all
    // (frame 1 is long before the lap line).
    expect(payloads[1]).not.toHaveProperty('rivalSwaps');

    // Every published event follows the operator exactly (numbering, seeds,
    // spawn-mind materialization, ledger compounding, JSON safety) and the
    // main agent is never a crossing anchor.
    assertLiveLaneSwapsFollowTheOperator(
      swaps,
      rivals,
      LEVEL_THIRD,
      LIVE_LANE_TRACK_SEED,
    );
  });

  it('stays a tested no-op at the lvl-2 rung in the live lane: the lone rival crosses and nobody is replaced', () => {
    const rivals = rivalCarIndicesForLevel(LEVEL_SECOND);
    const session = bootLiveLaneSession(
      LEVEL_SECOND,
      LIVE_LANE_AGENT_COUNT_LVL_SECOND,
    );

    // Act: drive until the lone rival crosses (or the budget is exhausted).
    const payloads = driveLiveLaneFrames(
      session,
      (swapsSoFar) => swapsSoFar >= 1,
    );
    const swaps = collectLiveLaneSwaps(payloads);

    // The drive is not vacuous: the lone rival really crossed.
    expect(swaps.length).toBeGreaterThanOrEqual(1);

    // The lvl-2 no-op in the live lane: empty propagated lists, and the lone
    // rival's ledger mind is never replaced — its crossing hex IS its spawn
    // mind at every event it anchors.
    const spawnMind = materializeLiveLaneSpawnMind(
      rivals[0],
      LEVEL_SECOND,
      LIVE_LANE_TRACK_SEED,
    );
    for (const payload of swaps) {
      expect(payload.propagatedCarIndices).toEqual([]);
      expect(payload.crossing.carIndex).toBe(rivals[0]);
      expect(payload.crossing.weights).toBe(hexOfSerializedWeights(spawnMind));
    }

    // The full operator mirror still holds for the no-op events (numbering,
    // seeds, the zero-replacement resolution, JSON safety).
    assertLiveLaneSwapsFollowTheOperator(
      swaps,
      rivals,
      LEVEL_SECOND,
      LIVE_LANE_TRACK_SEED,
    );
  });
});
