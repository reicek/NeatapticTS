import type { RacingCarState } from '../environment/environment.types';
import { generateTrack } from '../track/track.generator';
import type { TrackSpec } from '../track/track.generator.types';
import type { RacingObservationState } from './observation.assembler';

type Tier6ObservationState = RacingObservationState & {
  /** Pre-computed opponent slots, or omitted so the assembler can derive them from `envState.cars`. */
  opponentPerceptionSlots?: readonly (Float32Array | readonly number[])[];
};

interface Tier6ObservationAssemblerModule {
  derivePerCarObservationState(
    envState: RacingObservationState,
    carIndex: number,
  ): RacingObservationState;
  assembleTier6Observation(
    envState: Tier6ObservationState,
    trackSpec: TrackSpec,
  ): Float32Array | readonly number[];
  createTier6ObservationOptions(): {
    readonly tier: 6;
  };
}

const TIER6_OPPONENT_SLOT_COUNT = 3;
const TIER6_OPPONENT_CHANNELS_PER_SLOT = 7;
const TIER6_BASE_CHANNEL_COUNT = 103;
const TIER6_TOTAL_CHANNEL_COUNT = 124;
const TRACK_POSITION_WORLD_SCALE = 96;
const DISTANCE_WORLD_SCALE = 64;

/**
 * Red-phase contracts for the missing Tier 6 opponent-perception seam.
 *
 * Tier 6 extends the 103-channel Tier 5 vector with three opponent slots
 * (21 channels total), producing a 124-channel observation. Each slot is
 * expressed in the focal car's body frame and must stay in deterministic roster
 * order, never distance-sorted.
 */
describe('observation assembler Tier 6 opponent perception seam', () => {
  describe('ObservationTier type', () => {
    it('accepts 6 as a valid ObservationTier value', async () => {
      // Arrange
      const modulePromise = loadTier6ObservationAssemblerModule();

      // Act + Assert
      await expect(
        modulePromise.then(({ createTier6ObservationOptions }) =>
          acceptTier6ObservationTier(createTier6ObservationOptions().tier),
        ),
      ).resolves.toBe(6);
    });
  });

  describe('assembleTier6Observation', () => {
    it('returns exactly 124 channels', async () => {
      // Arrange
      const envState = createTier6ObservationState({
        cars: buildThreeOpponentRoster(),
      });
      const trackSpec = createTrackSpec();
      const modulePromise = loadTier6ObservationAssemblerModule();

      // Act + Assert
      await expect(
        modulePromise.then(
          ({ derivePerCarObservationState, assembleTier6Observation }) => {
            const derivedState = derivePerCarObservationState(envState, 0);
            return Array.from(assembleTier6Observation(derivedState, trackSpec))
              .length;
          },
        ),
      ).resolves.toBe(TIER6_TOTAL_CHANNEL_COUNT);
    });

    it('allocates exactly 7 channels per opponent slot', async () => {
      // Arrange
      const envState = createTier6ObservationState({
        cars: buildThreeOpponentRoster(),
      });
      const trackSpec = createTrackSpec();
      const modulePromise = loadTier6ObservationAssemblerModule();

      // Act + Assert
      await expect(
        modulePromise.then(
          ({ derivePerCarObservationState, assembleTier6Observation }) => {
            const derivedState = derivePerCarObservationState(envState, 0);
            const observationVector = Array.from(
              assembleTier6Observation(derivedState, trackSpec),
            );
            return [
              observationVector.slice(
                TIER6_BASE_CHANNEL_COUNT,
                TIER6_BASE_CHANNEL_COUNT + TIER6_OPPONENT_CHANNELS_PER_SLOT,
              ).length,
              observationVector.slice(
                TIER6_BASE_CHANNEL_COUNT + TIER6_OPPONENT_CHANNELS_PER_SLOT,
                TIER6_BASE_CHANNEL_COUNT + 2 * TIER6_OPPONENT_CHANNELS_PER_SLOT,
              ).length,
              observationVector.slice(
                TIER6_BASE_CHANNEL_COUNT + 2 * TIER6_OPPONENT_CHANNELS_PER_SLOT,
                TIER6_TOTAL_CHANNEL_COUNT,
              ).length,
            ];
          },
        ),
      ).resolves.toEqual([
        TIER6_OPPONENT_CHANNELS_PER_SLOT,
        TIER6_OPPONENT_CHANNELS_PER_SLOT,
        TIER6_OPPONENT_CHANNELS_PER_SLOT,
      ]);
    });

    it('zero-pads opponent slots 1 and 2 when only one opponent exists', async () => {
      // Arrange
      const envState = createTier6ObservationState({
        cars: [
          createCarState({
            x: 0,
            y: 0,
            heading: 0,
            teamIndex: 0,
          }),
          createCarState({
            x: 96,
            y: 0,
            heading: 0,
            teamIndex: 1,
          }),
        ],
      });
      const trackSpec = createTrackSpec();
      const modulePromise = loadTier6ObservationAssemblerModule();

      // Act + Assert
      await expect(
        modulePromise.then(
          ({ derivePerCarObservationState, assembleTier6Observation }) => {
            const derivedState = derivePerCarObservationState(envState, 0);
            const observationVector = Array.from(
              assembleTier6Observation(derivedState, trackSpec),
            );
            return observationVector.slice(
              TIER6_BASE_CHANNEL_COUNT + TIER6_OPPONENT_CHANNELS_PER_SLOT,
              TIER6_TOTAL_CHANNEL_COUNT,
            );
          },
        ),
      ).resolves.toEqual(
        new Array<number>(
          (TIER6_OPPONENT_SLOT_COUNT - 1) * TIER6_OPPONENT_CHANNELS_PER_SLOT,
        ).fill(0),
      );
    });

    it('computes relForwardEgo and relLeftEgo from the focal body frame', async () => {
      // Arrange
      // Focal car at origin, heading east (+X). Opponent 96 units ahead.
      const envState = createTier6ObservationState({
        cars: [
          createCarState({
            x: 0,
            y: 0,
            heading: 0,
            teamIndex: 0,
          }),
          createCarState({
            x: 96,
            y: 0,
            heading: Math.PI / 2,
            teamIndex: 1,
          }),
        ],
      });
      const trackSpec = createTrackSpec();
      const modulePromise = loadTier6ObservationAssemblerModule();

      // Act + Assert
      // Slot 0 channels 0..1 should be [96/96, 0].
      await expect(
        modulePromise.then(
          ({ derivePerCarObservationState, assembleTier6Observation }) => {
            const derivedState = derivePerCarObservationState(envState, 0);
            const observationVector = Array.from(
              assembleTier6Observation(derivedState, trackSpec),
            );
            return observationVector.slice(
              TIER6_BASE_CHANNEL_COUNT,
              TIER6_BASE_CHANNEL_COUNT + 2,
            );
          },
        ),
      ).resolves.toEqual([1, 0]);
    });

    it('computes sinHeadingDeltaEgo and cosHeadingDeltaEgo for known headings', async () => {
      // Arrange
      // Focal heading 0, opponent heading pi/3.
      const envState = createTier6ObservationState({
        cars: [
          createCarState({
            x: 0,
            y: 0,
            heading: 0,
            teamIndex: 0,
          }),
          createCarState({
            x: 48,
            y: 0,
            heading: Math.PI / 3,
            teamIndex: 1,
          }),
        ],
      });
      const trackSpec = createTrackSpec();
      const modulePromise = loadTier6ObservationAssemblerModule();

      // Act + Assert
      await expect(
        modulePromise.then(
          ({ derivePerCarObservationState, assembleTier6Observation }) => {
            const derivedState = derivePerCarObservationState(envState, 0);
            const observationVector = Array.from(
              assembleTier6Observation(derivedState, trackSpec),
            );
            const sinDelta = observationVector[TIER6_BASE_CHANNEL_COUNT + 2];
            const cosDelta = observationVector[TIER6_BASE_CHANNEL_COUNT + 3];
            return [sinDelta, cosDelta];
          },
        ),
      ).resolves.toEqual([
        expect.closeTo(Math.sin(Math.PI / 3), 6),
        expect.closeTo(Math.cos(Math.PI / 3), 6),
      ]);
    });

    it('encodes directDistance as Euclidean distance normalized by 64', async () => {
      // Arrange
      // Opponent is 64 world units directly north of the focal car.
      const envState = createTier6ObservationState({
        cars: [
          createCarState({
            x: 0,
            y: 0,
            heading: 0,
            teamIndex: 0,
          }),
          createCarState({
            x: 0,
            y: DISTANCE_WORLD_SCALE,
            heading: 0,
            teamIndex: 1,
          }),
        ],
      });
      const trackSpec = createTrackSpec();
      const modulePromise = loadTier6ObservationAssemblerModule();

      // Act + Assert
      // Channel 6 of opponent slot 0 should equal 64/64 = 1.
      await expect(
        modulePromise.then(
          ({ derivePerCarObservationState, assembleTier6Observation }) => {
            const derivedState = derivePerCarObservationState(envState, 0);
            const observationVector = Array.from(
              assembleTier6Observation(derivedState, trackSpec),
            );
            return observationVector[
              TIER6_BASE_CHANNEL_COUNT + TIER6_OPPONENT_CHANNELS_PER_SLOT - 1
            ];
          },
        ),
      ).resolves.toBeCloseTo(1, 6);
    });

    it('uses deterministic roster order, not distance-sorted order', async () => {
      // Arrange
      // Three opponents on the other team. The closest opponent is second in
      // the roster, the farthest is third. A distance-sorted layout would reorder
      // them; the contract keeps roster order.
      const envState = createTier6ObservationState({
        cars: [
          createCarState({
            x: 0,
            y: 0,
            heading: 0,
            teamIndex: 0,
          }),
          createCarState({
            x: 96,
            y: 0,
            heading: 0,
            teamIndex: 1,
          }),
          createCarState({
            x: 48,
            y: 0,
            heading: 0,
            teamIndex: 1,
          }),
          createCarState({
            x: 192,
            y: 0,
            heading: 0,
            teamIndex: 1,
          }),
        ],
      });
      const trackSpec = createTrackSpec();
      const modulePromise = loadTier6ObservationAssemblerModule();

      // Act + Assert
      // relForwardEgo is the first channel of each opponent slot.
      await expect(
        modulePromise.then(
          ({ derivePerCarObservationState, assembleTier6Observation }) => {
            const derivedState = derivePerCarObservationState(envState, 0);
            const observationVector = Array.from(
              assembleTier6Observation(derivedState, trackSpec),
            );
            return [
              observationVector[TIER6_BASE_CHANNEL_COUNT],
              observationVector[
                TIER6_BASE_CHANNEL_COUNT + TIER6_OPPONENT_CHANNELS_PER_SLOT
              ],
              observationVector[
                TIER6_BASE_CHANNEL_COUNT + 2 * TIER6_OPPONENT_CHANNELS_PER_SLOT
              ],
            ];
          },
        ),
      ).resolves.toEqual([
        96 / TRACK_POSITION_WORLD_SCALE,
        48 / TRACK_POSITION_WORLD_SCALE,
        192 / TRACK_POSITION_WORLD_SCALE,
      ]);
    });

    it('filters out same-team cars in a 3v3 mixed roster', async () => {
      // Arrange
      const envState = createTier6ObservationState({
        cars: buildMixedTeamRoster(),
      });
      const trackSpec = createTrackSpec();
      const modulePromise = loadTier6ObservationAssemblerModule();

      // Act + Assert
      // The three opponent slots should contain only the three Team B cars in
      // roster order, ignoring the two Team A teammates.
      await expect(
        modulePromise.then(
          ({ derivePerCarObservationState, assembleTier6Observation }) => {
            const derivedState = derivePerCarObservationState(envState, 0);
            const observationVector = Array.from(
              assembleTier6Observation(derivedState, trackSpec),
            );
            return [
              observationVector[TIER6_BASE_CHANNEL_COUNT],
              observationVector[
                TIER6_BASE_CHANNEL_COUNT + TIER6_OPPONENT_CHANNELS_PER_SLOT
              ],
              observationVector[
                TIER6_BASE_CHANNEL_COUNT + 2 * TIER6_OPPONENT_CHANNELS_PER_SLOT
              ],
            ];
          },
        ),
      ).resolves.toEqual([
        96 / TRACK_POSITION_WORLD_SCALE,
        48 / TRACK_POSITION_WORLD_SCALE,
        192 / TRACK_POSITION_WORLD_SCALE,
      ]);
    });

    it('projects body-frame speeds into the focal body frame', async () => {
      // Arrange
      // Focal car drives east at 54 world-units/second. Opponent drives north at
      // 54 world-units/second. Relative world velocity = (-54, 54), which
      // projects to -0.5 forward and +1.0 left after normalization.
      const SPEED_WORLD_SCALE = 108;
      const LATERAL_SPEED_WORLD_SCALE = 54;
      const envState = createTier6ObservationState({
        cars: [
          createCarState({
            x: 0,
            y: 0,
            heading: 0,
            teamIndex: 0,
            forwardSpeedWorld: SPEED_WORLD_SCALE / 2,
            lateralSpeedWorld: 0,
          }),
          createCarState({
            x: 0,
            y: 0,
            heading: Math.PI / 2,
            teamIndex: 1,
            forwardSpeedWorld: LATERAL_SPEED_WORLD_SCALE,
            lateralSpeedWorld: 0,
          }),
        ],
      });
      const trackSpec = createTrackSpec();
      const modulePromise = loadTier6ObservationAssemblerModule();

      // Act + Assert
      await expect(
        modulePromise.then(
          ({ derivePerCarObservationState, assembleTier6Observation }) => {
            const derivedState = derivePerCarObservationState(envState, 0);
            const observationVector = Array.from(
              assembleTier6Observation(derivedState, trackSpec),
            );
            const slotStart = TIER6_BASE_CHANNEL_COUNT;
            return [
              observationVector[slotStart + 4],
              observationVector[slotStart + 5],
            ];
          },
        ),
      ).resolves.toEqual([expect.closeTo(-0.5, 6), expect.closeTo(1, 6)]);
    });
  });
});

/**
 * Builds a focal Team A car plus three Team B opponents in roster order.
 */
function buildThreeOpponentRoster(): readonly RacingCarState[] {
  return [
    createCarState({
      x: 0,
      y: 0,
      heading: 0,
      teamIndex: 0,
    }),
    createCarState({
      x: 96,
      y: 0,
      heading: 0,
      teamIndex: 1,
    }),
    createCarState({
      x: 48,
      y: 0,
      heading: 0,
      teamIndex: 1,
    }),
    createCarState({
      x: 192,
      y: 0,
      heading: 0,
      teamIndex: 1,
    }),
  ];
}

/**
 * Builds a 3v3 mixed roster: one focal Team A car plus two Team A teammates and
 * three Team B opponents. Team B cars are interleaved in roster order to prove
 * that filtering uses teamIndex, not roster index.
 */
function buildMixedTeamRoster(): readonly RacingCarState[] {
  return [
    createCarState({
      x: 0,
      y: 0,
      heading: 0,
      teamIndex: 0,
    }),
    createCarState({
      x: 96,
      y: 0,
      heading: 0,
      teamIndex: 1,
    }),
    createCarState({
      x: 1_000,
      y: 0,
      heading: 0,
      teamIndex: 0,
    }),
    createCarState({
      x: 48,
      y: 0,
      heading: 0,
      teamIndex: 1,
    }),
    createCarState({
      x: 2_000,
      y: 0,
      heading: 0,
      teamIndex: 0,
    }),
    createCarState({
      x: 192,
      y: 0,
      heading: 0,
      teamIndex: 1,
    }),
  ];
}

/**
 * Creates a minimal car state for deterministic opponent-perception fixtures.
 */
function createCarState(options: {
  readonly x: number;
  readonly y: number;
  readonly heading: number;
  readonly teamIndex: 0 | 1;
  readonly forwardSpeedWorld?: number;
  readonly lateralSpeedWorld?: number;
}): RacingCarState {
  const speedWorld = Math.abs(options.forwardSpeedWorld ?? 0);
  return {
    carX: options.x,
    carY: options.y,
    carHeading: options.heading,
    teamIndex: options.teamIndex,
    tireState: [1, 1, 1, 1],
    forwardSpeedWorld: options.forwardSpeedWorld ?? 0,
    lateralSpeedWorld: options.lateralSpeedWorld ?? 0,
    speedWorld,
  };
}

function createTier6ObservationState(
  overrides: Partial<Tier6ObservationState> = {},
): Tier6ObservationState {
  const defaultState: Tier6ObservationState = {
    tick: 21,
    carX: 0,
    carY: 0,
    carHeading: 0,
    forwardSpeedWorld: 0,
    lateralSpeedWorld: 0,
    speedWorld: 0,
    yawRateRadiansPerSecond: 0,
    slipAngleRadians: 0,
    progress01: 0,
    lapProgress01: 0,
    boundaryDistanceLeftWorld: 14,
    boundaryDistanceRightWorld: 9,
    hazardDistanceWorld: 32,
    waypointDistanceWorld: 21,
    optimalLineLateralOffsetWorld: -3.5,
    optimalLineHeadingErrorRadians: 0,
    targetSpeedWorld: 24,
    memoryTrace: [0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4, 0.5, -0.5],
    teammateRadioSlots: [],
    tireState: [0.95, 0.9, 0.85, 0.8],
  };

  return {
    ...defaultState,
    ...overrides,
    teammateRadioSlots:
      overrides.teammateRadioSlots ?? defaultState.teammateRadioSlots,
    tireState: overrides.tireState ?? defaultState.tireState,
  };
}

function createTrackSpec(): TrackSpec {
  return generateTrack({ seed: 42, layoutVersion: 1, sizeBucket: 'medium' });
}

function acceptTier6ObservationTier(
  tier: 1 | 2 | 3 | 4 | 5 | 6,
): 1 | 2 | 3 | 4 | 5 | 6 {
  return tier;
}

async function loadTier6ObservationAssemblerModule(): Promise<Tier6ObservationAssemblerModule> {
  const modulePath = './observation.assembler';
  const module = (await import(
    modulePath
  )) as Partial<Tier6ObservationAssemblerModule>;

  if (typeof module.assembleTier6Observation !== 'function') {
    throw new Error(
      'Missing Tier 6 observation export: assembleTier6Observation',
    );
  }

  if (typeof module.createTier6ObservationOptions !== 'function') {
    throw new Error(
      'Missing Tier 6 observation export: createTier6ObservationOptions',
    );
  }

  return module as Tier6ObservationAssemblerModule;
}
