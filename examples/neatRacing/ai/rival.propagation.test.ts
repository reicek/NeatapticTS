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
 * @module
 */

import {
  serializeNeatRacingReproducibilityTuple,
  createNeatRacingXorShift32Stream,
  deriveNeatRacingRivalSwapChildSeed,
  NEATRACING_RIVAL_SWAP_SEED_STRIDE,
  type NeatRacingReproducibilityTuple,
} from '../workers/simulation-worker/reproducibility.utils';
import { NEATRACING_TEAM_LAYOUTS } from './observation.assembler';
import {
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
  const layout = NEATRACING_TEAM_LAYOUTS[
    level as keyof typeof NEATRACING_TEAM_LAYOUTS
  ];
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

  it('keeps every replacement on the winner\'s flat weight plane', () => {
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

  it('never mutates the winner\'s serialized weights', () => {
    const winnerWeights = new Float32Array([1, -0.5, 0.25, 2]);
    propagateNeatRacingRivalWinnerWeights(
      buildSwapOptions({ winnerWeights }),
    );
    expect(winnerWeights).toEqual(new Float32Array([1, -0.5, 0.25, 2]));
  });
});

// ── The main-agent exclusion ─────────────────────────────────────────────────

describe('the main-agent exclusion (the Blocked stop condition is a guard)', () => {
  it('rejects a rival set that includes the main agent\'s car 0 slot', () => {
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
    const { snapshot } = propagateNeatRacingRivalWinnerWeights(
      buildSwapOptions(),
    );
    expect(snapshot.crossing.carIndex).toBe(2);
    expect(snapshot.eventIndex).toBe(EVENT_FIRST);
  });

  it('carries the derived mutation seed', () => {
    const { snapshot } = propagateNeatRacingRivalWinnerWeights(
      buildSwapOptions(),
    );
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
    const { snapshot } = propagateNeatRacingRivalWinnerWeights(
      buildSwapOptions(),
    );
    const payload = toNeatRacingRivalSwapPayload(snapshot);
    expect(payload.crossing.weights).toBe(WINNER_WEIGHTS_HEX);
  });

  it('carries the discriminator and the event fields', () => {
    const { snapshot } = propagateNeatRacingRivalWinnerWeights(
      buildSwapOptions(),
    );
    const payload = toNeatRacingRivalSwapPayload(snapshot);
    expect(payload.kind).toBe('rival-swap');
    expect(payload.eventIndex).toBe(EVENT_FIRST);
    expect(payload.mutationSeed).toBe(FIRST_SWAP_MUTATION_SEED);
  });

  it('survives the ONE base tuple serial path', () => {
    const { snapshot } = propagateNeatRacingRivalWinnerWeights(
      buildSwapOptions(),
    );
    const payload = toNeatRacingRivalSwapPayload(snapshot);
    const tuple = buildBaseTupleFixture(payload);
    const parsed = JSON.parse(
      serializeNeatRacingReproducibilityTuple(tuple),
    ) as { opponentSnapshot: NeatRacingRivalSwapPayload };
    expect(parsed.opponentSnapshot).toEqual(payload);
  });

  it('rejects a payload whose weights field carries a typed array', () => {
    const { snapshot } = propagateNeatRacingRivalWinnerWeights(
      buildSwapOptions(),
    );
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
    const { snapshot, replacements } = propagateNeatRacingRivalWinnerWeights(
      buildSwapOptions(),
    );
    const payload = toNeatRacingRivalSwapPayload(snapshot);
    expect(restoreNeatRacingRivalSwapReplacementMinds(payload)).toEqual(
      replacements,
    );
  });

  it('is replay-exact across repeated restores', () => {
    const { snapshot } = propagateNeatRacingRivalWinnerWeights(
      buildSwapOptions(),
    );
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
    const firstEvent = propagateNeatRacingRivalWinnerWeights(buildSwapOptions());
    const secondEvent = propagateNeatRacingRivalWinnerWeights(
      buildSwapOptions({ eventIndex: EVENT_SECOND }),
    );
    expect(secondEvent.replacements).not.toEqual(firstEvent.replacements);
  });

  it('round-trips through the canonical serial (serialize -> parse -> restore)', () => {
    const { snapshot, replacements } = propagateNeatRacingRivalWinnerWeights(
      buildSwapOptions(),
    );
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
    const { snapshot } = propagateNeatRacingRivalWinnerWeights(
      buildSwapOptions(),
    );
    const payload = toNeatRacingRivalSwapPayload(snapshot);
    expect(serializeNeatRacingRivalSwapPayload(payload)).toBe(
      serializeNeatRacingRivalSwapPayload(payload),
    );
  });

  it('is key-order invariant', () => {
    const { snapshot } = propagateNeatRacingRivalWinnerWeights(
      buildSwapOptions(),
    );
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
    const { snapshot } = propagateNeatRacingRivalWinnerWeights(
      buildSwapOptions(),
    );
    const payload = toNeatRacingRivalSwapPayload(snapshot);
    const serial = serializeNeatRacingRivalSwapPayload(payload);
    expect(serial).toContain('"kind":"rival-swap"');
    expect(serial).toContain(`"weights":"${WINNER_WEIGHTS_HEX}"`);
    expect(serial).toContain('"eventIndex":0');
  });

  it('rejects a NaN mutation seed', () => {
    const { snapshot } = propagateNeatRacingRivalWinnerWeights(
      buildSwapOptions(),
    );
    const payload = toNeatRacingRivalSwapPayload(snapshot);
    const tampered = { ...payload, mutationSeed: Number.NaN };
    expect(() => serializeNeatRacingRivalSwapPayload(tampered)).toThrow(
      RangeError,
    );
  });

  it('rejects an uppercase-hex weights field', () => {
    const { snapshot } = propagateNeatRacingRivalWinnerWeights(
      buildSwapOptions(),
    );
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
    const { snapshot } = propagateNeatRacingRivalWinnerWeights(
      buildSwapOptions(),
    );
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
    const { snapshot } = propagateNeatRacingRivalWinnerWeights(
      buildSwapOptions(),
    );
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
    const { snapshot } = propagateNeatRacingRivalWinnerWeights(
      buildSwapOptions(),
    );
    expect(() =>
      toNeatRacingRivalSwapPayload({
        ...snapshot,
        mutationSeed: 0x100000000,
      } as unknown as NeatRacingRivalSwapSnapshot),
    ).toThrow(RangeError);
  });

  it('rejects a main-agent-slot crossing index', () => {
    const { snapshot } = propagateNeatRacingRivalWinnerWeights(
      buildSwapOptions(),
    );
    expect(() =>
      toNeatRacingRivalSwapPayload({
        ...snapshot,
        crossing: { ...snapshot.crossing, carIndex: 0 },
      } as unknown as NeatRacingRivalSwapSnapshot),
    ).toThrow(RangeError);
  });

  it('rejects empty or non-finite crossing weights', () => {
    const { snapshot } = propagateNeatRacingRivalWinnerWeights(
      buildSwapOptions(),
    );
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
    const { snapshot } = propagateNeatRacingRivalWinnerWeights(
      buildSwapOptions(),
    );
    expect(() =>
      toNeatRacingRivalSwapPayload({
        ...snapshot,
        propagatedCarIndices: [3, 1],
      } as unknown as NeatRacingRivalSwapSnapshot),
    ).toThrow(RangeError);
  });

  it('rejects propagated indices carrying the crossing rival', () => {
    const { snapshot } = propagateNeatRacingRivalWinnerWeights(
      buildSwapOptions(),
    );
    expect(() =>
      toNeatRacingRivalSwapPayload({
        ...snapshot,
        propagatedCarIndices: [2, 3],
      } as unknown as NeatRacingRivalSwapSnapshot),
    ).toThrow(RangeError);
  });

  it('rejects a snapshot record missing its crossing snapshot', () => {
    const { snapshot } = propagateNeatRacingRivalWinnerWeights(
      buildSwapOptions(),
    );
    expect(() =>
      toNeatRacingRivalSwapPayload({
        ...snapshot,
        crossing: undefined,
      } as unknown as NeatRacingRivalSwapSnapshot),
    ).toThrow(RangeError);
  });

  it('rejects a payload record missing its crossing snapshot', () => {
    const { snapshot } = propagateNeatRacingRivalWinnerWeights(
      buildSwapOptions(),
    );
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
    const { replacements } = propagateNeatRacingRivalWinnerWeights(
      buildSwapOptions(),
    );
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
    const { snapshot } = propagateNeatRacingRivalWinnerWeights(
      buildSwapOptions(),
    );
    const payload = toNeatRacingRivalSwapPayload(snapshot);
    const nanHexPayload = {
      ...payload,
      crossing: { ...payload.crossing, weights: '0000c07f' },
    } as unknown as NeatRacingRivalSwapPayload;
    expect(() =>
      serializeNeatRacingRivalSwapPayload(nanHexPayload),
    ).toThrow(RangeError);
    expect(() =>
      restoreNeatRacingRivalSwapReplacementMinds(nanHexPayload),
    ).toThrow(RangeError);
  });
});
