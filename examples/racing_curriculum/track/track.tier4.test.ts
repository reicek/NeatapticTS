import type {
  TrackGeneratorInput,
  TrackSpec,
  TrackSegment,
} from './track.generator.types';

type PitCorridorAabb = {
  x: number;
  y: number;
  width: number;
  height: number;
};
type PitBoxSpec = {
  teamIndex: number;
  entranceCorridor: PitCorridorAabb;
};
type Tier4TrackSpec = TrackSpec & {
  pitBoxes: readonly [PitBoxSpec, PitBoxSpec];
};

interface Tier4TrackModule {
  generateTrack(input: TrackGeneratorInput): Tier4TrackSpec;
  validateTrackSpec(spec: Tier4TrackSpec): true;
  validatePitCorridorNonOverlap(spec: Tier4TrackSpec): true;
  validatePitCorridorReachability(spec: Tier4TrackSpec): true;
}

describe('track Tier 4 pit geometry seam', () => {
  describe('generateTrack', () => {
    it('returns two pit boxes ordered by team index 0 then 1', async () => {
      // Arrange
      const generatorInput: TrackGeneratorInput = {
        seed: 42,
        layoutVersion: 1,
        sizeBucket: 'medium',
      };

      // Act + Assert
      await expect(
        loadTier4TrackModule().then(({ generateTrack }) =>
          generateTrack(generatorInput).pitBoxes.map(
            ({ teamIndex }) => teamIndex,
          ),
        ),
      ).resolves.toEqual([0, 1]);
    });

    it('attaches an entrance corridor AABB to every pit box', async () => {
      // Arrange
      const generatorInput: TrackGeneratorInput = {
        seed: 84,
        layoutVersion: 1,
        sizeBucket: 'medium',
      };

      // Act + Assert
      await expect(
        loadTier4TrackModule().then(({ generateTrack }) =>
          generateTrack(generatorInput).pitBoxes.every(
            ({ entranceCorridor }) =>
              entranceCorridor.width > 0 &&
              entranceCorridor.height > 0 &&
              Number.isFinite(entranceCorridor.x) &&
              Number.isFinite(entranceCorridor.y),
          ),
        ),
      ).resolves.toBe(true);
    });
  });

  describe('pit corridor validation', () => {
    it('rejects overlapping pit entrance corridors', async () => {
      // Arrange
      const overlappingTrackSpec = createTier4TrackSpec({
        pitBoxes: [
          {
            teamIndex: 0,
            entranceCorridor: { x: 20, y: 20, width: 12, height: 12 },
          },
          {
            teamIndex: 1,
            entranceCorridor: { x: 24, y: 24, width: 12, height: 12 },
          },
        ],
      });

      // Act + Assert
      await expect(
        loadTier4TrackModule().then(({ validatePitCorridorNonOverlap }) => {
          try {
            validatePitCorridorNonOverlap(overlappingTrackSpec);
            return false;
          } catch (error) {
            return error instanceof RangeError;
          }
        }),
      ).resolves.toBe(true);
    });

    it('rejects a pit corridor that is unreachable from the track path', async () => {
      // Arrange
      const unreachableTrackSpec = createTier4TrackSpec({
        pitBoxes: [
          {
            teamIndex: 0,
            entranceCorridor: { x: 400, y: 400, width: 10, height: 10 },
          },
          {
            teamIndex: 1,
            entranceCorridor: { x: 120, y: 20, width: 10, height: 10 },
          },
        ],
      });

      // Act + Assert
      await expect(
        loadTier4TrackModule().then(({ validatePitCorridorReachability }) => {
          try {
            validatePitCorridorReachability(unreachableTrackSpec);
            return false;
          } catch (error) {
            return error instanceof RangeError;
          }
        }),
      ).resolves.toBe(true);
    });
  });
});

function createTier4TrackSpec(
  overrides: Partial<Tier4TrackSpec> = {},
): Tier4TrackSpec {
  const segments: readonly TrackSegment[] = [
    { startX: 0, startY: 0, endX: 120, endY: 0, width: 20 },
    { startX: 120, startY: 0, endX: 120, endY: 120, width: 20 },
    { startX: 120, startY: 120, endX: 0, endY: 120, width: 20 },
    { startX: 0, startY: 120, endX: 0, endY: 0, width: 20 },
  ];
  const defaultTrackSpec: Tier4TrackSpec = {
    seed: 7,
    layoutVersion: 1,
    sizeBucket: 'unit-test',
    segments,
    splineSamples: [],
    pitBoxes: [
      {
        teamIndex: 0,
        entranceCorridor: { x: 12, y: 4, width: 10, height: 12 },
      },
      {
        teamIndex: 1,
        entranceCorridor: { x: 98, y: 4, width: 10, height: 12 },
      },
    ],
  };

  return {
    ...defaultTrackSpec,
    ...overrides,
    pitBoxes: overrides.pitBoxes ?? defaultTrackSpec.pitBoxes,
  };
}

async function loadTier4TrackModule(): Promise<Tier4TrackModule> {
  const generatorModule = await import('./track.generator');
  const validationModule = await import('./track.validation');
  const tier4ValidationModule = validationModule as Partial<Tier4TrackModule>;

  if (
    typeof tier4ValidationModule.validatePitCorridorNonOverlap !== 'function'
  ) {
    throw new Error(
      'Missing Tier 4 track validation export: validatePitCorridorNonOverlap',
    );
  }

  if (
    typeof tier4ValidationModule.validatePitCorridorReachability !== 'function'
  ) {
    throw new Error(
      'Missing Tier 4 track validation export: validatePitCorridorReachability',
    );
  }

  return {
    generateTrack:
      generatorModule.generateTrack as Tier4TrackModule['generateTrack'],
    validateTrackSpec:
      validationModule.validateTrackSpec as Tier4TrackModule['validateTrackSpec'],
    validatePitCorridorNonOverlap:
      tier4ValidationModule.validatePitCorridorNonOverlap as Tier4TrackModule['validatePitCorridorNonOverlap'],
    validatePitCorridorReachability:
      tier4ValidationModule.validatePitCorridorReachability as Tier4TrackModule['validatePitCorridorReachability'],
  };
}
