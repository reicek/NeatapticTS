import type { TrackSpec } from '../track/track.generator.types';

type TireStateTuple = readonly [number, number, number, number];
type PitStatusRecord = {
  occupyingCarIndex: number;
  remainingStopTicks: number;
};
type PitCorridorAabb = {
  x: number;
  y: number;
  width: number;
  height: number;
};
type PitBoxSpec = {
  teamIndex: number;
  pitBox: PitCorridorAabb;
  entranceCorridor: PitCorridorAabb;
};
type Tier4TrackSpec = TrackSpec & {
  pitBoxes: readonly [PitBoxSpec, PitBoxSpec];
};
type Tier4CarState = {
  carX: number;
  carY: number;
  carHeading: number;
  teamIndex: number;
  tireState: TireStateTuple;
};
type Tier4EnvironmentState = {
  tick: number;
  cars: readonly Tier4CarState[];
  pitStatus: readonly [PitStatusRecord, PitStatusRecord];
  trackSpec: Tier4TrackSpec;
};
type CarControlOutput = {
  throttle: number;
  steer: number;
};

interface Tier4EnvironmentModule {
  createInitialState(): Tier4EnvironmentState;
  stepEnvironment(
    state: Tier4EnvironmentState,
    controls: readonly CarControlOutput[],
  ): Tier4EnvironmentState;
  decayTireState(
    current: TireStateTuple,
    lateralForce: number,
    longitudinalForce: number,
    speed: number,
  ): TireStateTuple;
}

describe('environment Tier 4 tire and pit seam', () => {
  describe('decayTireState', () => {
    it('exists and reduces tire state from 1.0 under non-zero forces', async () => {
      // Arrange
      const freshTireState: TireStateTuple = [1, 1, 1, 1];

      // Act + Assert
      await expect(
        loadTier4EnvironmentModule().then(({ decayTireState }) =>
          decayTireState(freshTireState, 2.4, 1.8, 32).every(
            (tireStateValue) => tireStateValue < 1,
          ),
        ),
      ).resolves.toBe(true);
    });

    it('keeps zero-force decay at minimal or rest-decay-only levels', async () => {
      // Arrange
      const freshTireState: TireStateTuple = [1, 1, 1, 1];

      // Act + Assert
      await expect(
        loadTier4EnvironmentModule().then(({ decayTireState }) =>
          decayTireState(freshTireState, 0, 0, 0).every(
            (tireStateValue) => tireStateValue <= 1 && tireStateValue >= 0.99,
          ),
        ),
      ).resolves.toBe(true);
    });

    it('clamps the result to the closed [0.0, 1.0] interval', async () => {
      // Arrange
      const stressedTireState: TireStateTuple = [0.02, 0.25, 0.9, 1];

      // Act + Assert
      await expect(
        loadTier4EnvironmentModule().then(({ decayTireState }) =>
          decayTireState(stressedTireState, 20, 20, 80).every(
            (tireStateValue) => tireStateValue >= 0 && tireStateValue <= 1,
          ),
        ),
      ).resolves.toBe(true);
    });

    it('degrades half-worn tires faster than fresh tires at the same load', async () => {
      // Arrange
      const freshTireState: TireStateTuple = [1, 1, 1, 1];
      const halfWornTireState: TireStateTuple = [0.5, 0.5, 0.5, 0.5];

      // Act + Assert
      await expect(
        loadTier4EnvironmentModule().then(({ decayTireState }) => {
          const freshDecay = decayTireState(freshTireState, 2.2, 1.4, 28);
          const halfWornDecay = decayTireState(halfWornTireState, 2.2, 1.4, 28);
          const freshLoss = freshDecay.reduce(
            (lossAccumulator, tireStateValue) =>
              lossAccumulator + (1 - tireStateValue),
            0,
          );
          const halfWornLoss = halfWornDecay.reduce(
            (lossAccumulator, tireStateValue) =>
              lossAccumulator + (0.5 - tireStateValue),
            0,
          );
          return halfWornLoss > freshLoss;
        }),
      ).resolves.toBe(true);
    });

    it('keeps fresh tires above the red threshold for about two representative laps', async () => {
      // Arrange
      const freshTireState: TireStateTuple = [1, 1, 1, 1];

      // Act + Assert
      await expect(
        loadTier4EnvironmentModule().then(({ decayTireState }) => {
          let nextTireState = freshTireState;

          for (let stepIndex = 0; stepIndex < 240; stepIndex += 1) {
            nextTireState = decayTireState(nextTireState, 2.2, 1.4, 28);
          }

          return nextTireState.every((tireStateValue) => tireStateValue > 0.5);
        }),
      ).resolves.toBe(true);
    });
  });

  describe('pit occupancy lifecycle', () => {
    it('starts each team with an empty pit occupancy record', async () => {
      // Arrange + Act + Assert
      await expect(
        loadTier4EnvironmentModule().then(
          ({ createInitialState }) => createInitialState().pitStatus,
        ),
      ).resolves.toEqual([
        { occupyingCarIndex: 255, remainingStopTicks: 0 },
        { occupyingCarIndex: 255, remainingStopTicks: 0 },
      ]);
    });

    it('sets pit occupancy and starts the countdown when a car enters its pit box', async () => {
      // Arrange
      const tier4EnvironmentState = createTier4EnvironmentState({
        cars: [
          createTier4CarState({ carX: 12, carY: 12, teamIndex: 0 }),
          createTier4CarState({ carX: 40, carY: 40, teamIndex: 0 }),
          createTier4CarState({ carX: 120, carY: 120, teamIndex: 1 }),
          createTier4CarState({ carX: 160, carY: 160, teamIndex: 1 }),
        ],
      });

      // Act + Assert
      await expect(
        loadTier4EnvironmentModule().then(({ stepEnvironment }) => {
          const nextState = stepEnvironment(
            tier4EnvironmentState,
            createNeutralControlOutputs(),
          );
          return {
            occupyingCarIndex: nextState.pitStatus[0].occupyingCarIndex,
            countdownStarted: nextState.pitStatus[0].remainingStopTicks > 0,
          };
        }),
      ).resolves.toEqual({
        occupyingCarIndex: 0,
        countdownStarted: true,
      });
    });

    it('restores tires to full health and clears occupancy when the stop countdown ends', async () => {
      // Arrange
      const tier4EnvironmentState = createTier4EnvironmentState({
        cars: [
          createTier4CarState({
            carX: 12,
            carY: 12,
            teamIndex: 0,
            tireState: [0.25, 0.35, 0.4, 0.45],
          }),
          createTier4CarState({ carX: 40, carY: 40, teamIndex: 0 }),
          createTier4CarState({ carX: 120, carY: 120, teamIndex: 1 }),
          createTier4CarState({ carX: 160, carY: 160, teamIndex: 1 }),
        ],
        pitStatus: [
          { occupyingCarIndex: 0, remainingStopTicks: 1 },
          { occupyingCarIndex: 255, remainingStopTicks: 0 },
        ],
      });

      // Act + Assert
      await expect(
        loadTier4EnvironmentModule().then(({ stepEnvironment }) => {
          const nextState = stepEnvironment(
            tier4EnvironmentState,
            createNeutralControlOutputs(),
          );
          return {
            pitStatus: nextState.pitStatus[0],
            tireState: nextState.cars[0]?.tireState,
          };
        }),
      ).resolves.toEqual({
        pitStatus: { occupyingCarIndex: 255, remainingStopTicks: 0 },
        tireState: [1, 1, 1, 1],
      });
    });

    it('advances every car when the stepper receives a per-car control array', async () => {
      // Arrange
      const tier4EnvironmentState = createTier4EnvironmentState({
        cars: [
          createTier4CarState({ teamIndex: 0 }),
          createTier4CarState({ teamIndex: 0, carX: 40, carY: 40 }),
          createTier4CarState({ teamIndex: 1, carX: 120, carY: 120 }),
          createTier4CarState({ teamIndex: 1, carX: 160, carY: 160 }),
        ],
      });

      // Act + Assert
      await expect(
        loadTier4EnvironmentModule().then(({ stepEnvironment }) => {
          const nextState = stepEnvironment(tier4EnvironmentState, [
            { throttle: 1, steer: 0 },
            { throttle: 1, steer: 0 },
            { throttle: 1, steer: 0 },
            { throttle: 1, steer: 0 },
          ]);

          return nextState.cars.every((carState, carIndex) => {
            const previousCarState = tier4EnvironmentState.cars[carIndex];
            return (
              carState.carX !== previousCarState.carX ||
              carState.carY !== previousCarState.carY
            );
          });
        }),
      ).resolves.toBe(true);
    });
  });
});

function createTier4EnvironmentState(
  overrides: Partial<Tier4EnvironmentState> = {},
): Tier4EnvironmentState {
  const defaultState: Tier4EnvironmentState = {
    tick: 0,
    cars: [
      createTier4CarState({ teamIndex: 0 }),
      createTier4CarState({ teamIndex: 0, carX: 40, carY: 40 }),
      createTier4CarState({ teamIndex: 1, carX: 120, carY: 120 }),
      createTier4CarState({ teamIndex: 1, carX: 160, carY: 160 }),
    ],
    pitStatus: [
      { occupyingCarIndex: 255, remainingStopTicks: 0 },
      { occupyingCarIndex: 255, remainingStopTicks: 0 },
    ],
    trackSpec: createTier4TrackSpec(),
  };

  return {
    ...defaultState,
    ...overrides,
    cars: overrides.cars ?? defaultState.cars,
    pitStatus: overrides.pitStatus ?? defaultState.pitStatus,
    trackSpec: overrides.trackSpec ?? defaultState.trackSpec,
  };
}

function createTier4CarState(
  overrides: Partial<Tier4CarState> = {},
): Tier4CarState {
  return {
    carX: 0,
    carY: 0,
    carHeading: 0,
    teamIndex: 0,
    ...overrides,
    tireState: overrides.tireState ?? [1, 1, 1, 1],
  };
}

function createTier4TrackSpec(): Tier4TrackSpec {
  return {
    seed: 42,
    layoutVersion: 1,
    sizeBucket: 'medium',
    segments: [],
    splineSamples: [],
    pitBoxes: [
      {
        teamIndex: 0,
        pitBox: { x: 10, y: 10, width: 8, height: 8 },
        entranceCorridor: { x: 6, y: 8, width: 12, height: 12 },
      },
      {
        teamIndex: 1,
        pitBox: { x: 110, y: 110, width: 8, height: 8 },
        entranceCorridor: { x: 106, y: 108, width: 12, height: 12 },
      },
    ],
  };
}

function createNeutralControlOutputs(): readonly CarControlOutput[] {
  return [
    { throttle: 0, steer: 0 },
    { throttle: 0, steer: 0 },
    { throttle: 0, steer: 0 },
    { throttle: 0, steer: 0 },
  ];
}

async function loadTier4EnvironmentModule(): Promise<Tier4EnvironmentModule> {
  const modulePath = './environment.step.service';
  const module = (await import(modulePath)) as Partial<Tier4EnvironmentModule>;

  if (typeof module.decayTireState !== 'function') {
    throw new Error('Missing Tier 4 environment export: decayTireState');
  }

  return module as Tier4EnvironmentModule;
}
