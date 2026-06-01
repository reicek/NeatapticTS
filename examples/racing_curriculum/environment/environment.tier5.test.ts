import {
  createInitialState,
  decayTireState,
  stepEnvironment,
} from './environment.step.service';
import type {
  CarControlOutput,
  EnvironmentState,
  RacingCarState,
} from './environment.types';
import type { TrackSpec } from '../track/track.generator.types';

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

type Tier5TrackSpec = TrackSpec & {
  pitBoxes: readonly PitBoxSpec[];
};

describe('environment Tier 5 six-car seam', () => {
  describe('stepEnvironment', () => {
    it('steps a six-car 3v3 roster with carTeam [0, 0, 0, 1, 1, 1] without error', () => {
      // Arrange
      const initialState = createInitialState();
      const tier5State = {
        ...initialState,
        trackSpec: createTier5TrackSpec(),
      } as EnvironmentState;
      const neutralControls = createNeutralControlOutputs(6);

      // Act
      const nextState = stepEnvironment(tier5State, neutralControls);
      const nextTeamLayout = (nextState.cars ?? []).map((car) => car.teamIndex);

      // Assert
      expect(nextTeamLayout).toEqual([0, 0, 0, 1, 1, 1]);
    });

    it('produces 24 tire values after one six-car step', () => {
      // Arrange
      const initialState = createInitialState();
      const tier5State = {
        ...initialState,
        trackSpec: createTier5TrackSpec(),
      } as EnvironmentState;
      const neutralControls = createNeutralControlOutputs(6);

      // Act
      const nextState = stepEnvironment(tier5State, neutralControls);
      const flattenedTireValues = (nextState.cars ?? []).flatMap(
        (car) => car.tireState,
      );

      // Assert
      expect(flattenedTireValues.length).toBe(24);
    });

    it('keeps pitOccupancy at six slots while the runtime widens to six cars', () => {
      // Arrange
      const initialState = createInitialState();
      const tier5State = {
        ...initialState,
        trackSpec: createTier5TrackSpec(),
      } as EnvironmentState;
      const neutralControls = createNeutralControlOutputs(6);

      // Act
      const nextState = stepEnvironment(tier5State, neutralControls);
      const tier5Summary = {
        carCount: nextState.cars?.length ?? 0,
        pitSlotCount:
          nextState.pitOccupancy?.length ?? nextState.pitStatus?.length ?? 0,
      };

      // Assert
      expect(tier5Summary).toEqual({ carCount: 6, pitSlotCount: 6 });
    });

    it('keeps first-car-wins pit entry when three Team A cars contest one pit slot', () => {
      // Arrange
      const competitionState = createTeamPitCompetitionState();

      // Act
      const nextState = stepEnvironment(
        competitionState,
        createNeutralControlOutputs(6),
      );
      const pitCompetitionSummary = {
        teamLayout: (nextState.cars ?? []).map((car) => car.teamIndex),
        teamAOccupiedCarIndices: (nextState.pitStatus ?? [])
          .slice(0, 3)
          .map((pitSlot) => pitSlot.occupyingCarIndex)
          .filter((carIndex) => carIndex !== 255)
          .toSorted((leftCarIndex, rightCarIndex) => leftCarIndex - rightCarIndex),
      };

      // Assert
      expect(pitCompetitionSummary).toEqual({
        teamLayout: [0, 0, 0, 1, 1, 1],
        teamAOccupiedCarIndices: [0],
      });
    });

    it('allows three simultaneous pit stops per team when each car reaches its own pit corridor', () => {
      // Arrange
      const fullOccupancyState = createFullPitOccupancyState();

      // Act
      const nextState = stepEnvironment(
        fullOccupancyState,
        createNeutralControlOutputs(6),
      );
      const pitSlots = nextState.pitStatus ?? [];
      const fullOccupancySummary = {
        teamAOccupiedCarIndices: pitSlots
          .slice(0, 3)
          .map((pitSlot) => pitSlot.occupyingCarIndex)
          .toSorted((leftCarIndex, rightCarIndex) => leftCarIndex - rightCarIndex),
        teamBOccupiedCarIndices: pitSlots
          .slice(3, 6)
          .map((pitSlot) => pitSlot.occupyingCarIndex)
          .toSorted((leftCarIndex, rightCarIndex) => leftCarIndex - rightCarIndex),
        allSlotsStartedCountdown: pitSlots.every(
          (pitSlot) => pitSlot.remainingStopTicks > 0,
        ),
      };

      // Assert
      expect(fullOccupancySummary).toEqual({
        teamAOccupiedCarIndices: [0, 1, 2],
        teamBOccupiedCarIndices: [3, 4, 5],
        allSlotsStartedCountdown: true,
      });
    });
  });

  describe('decayTireState', () => {
    it('produces 24 clamped tire values for the six-car Tier 5 roster', () => {
      // Arrange
      const initialState = createInitialState();
      const tier5Cars = initialState.cars ?? [];

      // Act
      const decayedTireValues = tier5Cars.flatMap((car) =>
        decayTireState(car.tireState, 2.4, 1.8, 32),
      );
      const tireDecaySummary = {
        carCount: tier5Cars.length,
        tireValueCount: decayedTireValues.length,
        allValuesClamped: decayedTireValues.every(
          (tireValue) => tireValue >= 0 && tireValue <= 1,
        ),
      };

      // Assert
      expect(tireDecaySummary).toEqual({
        carCount: 6,
        tireValueCount: 24,
        allValuesClamped: true,
      });
    });
  });
});

function createTeamPitCompetitionState(): EnvironmentState {
  const initialState = createInitialState();
  const cars = (initialState.cars ?? []).map((car, carIndex) =>
    createPitCompetitionCar(car, carIndex),
  );

  return {
    ...initialState,
    cars,
    trackSpec: createTier5TrackSpec(),
  };
}

function createFullPitOccupancyState(): EnvironmentState {
  const initialState = createInitialState();
  const cars = (initialState.cars ?? []).map((car, carIndex) => {
    const pitEntryPositions: readonly [readonly [number, number], readonly [number, number], readonly [number, number], readonly [number, number], readonly [number, number], readonly [number, number]] = [
      [12, 12],
      [32, 12],
      [52, 12],
      [112, 112],
      [132, 112],
      [152, 112],
    ];
    const [carX, carY] = pitEntryPositions[carIndex] ?? [car.carX, car.carY];

    return {
      ...car,
      carX,
      carY,
    };
  });

  return {
    ...initialState,
    cars,
    trackSpec: createTier5TrackSpec(),
  };
}

function createPitCompetitionCar(
  car: RacingCarState,
  carIndex: number,
): RacingCarState {
  if (carIndex <= 2) {
    return {
      ...car,
      carX: 12 + carIndex,
      carY: 12,
    };
  }

  return {
    ...car,
    carX: 140 + carIndex,
    carY: 140,
  };
}

function createTier5TrackSpec(): Tier5TrackSpec {
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
        teamIndex: 0,
        pitBox: { x: 30, y: 10, width: 8, height: 8 },
        entranceCorridor: { x: 26, y: 8, width: 12, height: 12 },
      },
      {
        teamIndex: 0,
        pitBox: { x: 50, y: 10, width: 8, height: 8 },
        entranceCorridor: { x: 46, y: 8, width: 12, height: 12 },
      },
      {
        teamIndex: 1,
        pitBox: { x: 110, y: 110, width: 8, height: 8 },
        entranceCorridor: { x: 106, y: 108, width: 12, height: 12 },
      },
      {
        teamIndex: 1,
        pitBox: { x: 130, y: 110, width: 8, height: 8 },
        entranceCorridor: { x: 126, y: 108, width: 12, height: 12 },
      },
      {
        teamIndex: 1,
        pitBox: { x: 150, y: 110, width: 8, height: 8 },
        entranceCorridor: { x: 146, y: 108, width: 12, height: 12 },
      },
    ],
  };
}

function createNeutralControlOutputs(
  carCount: number,
): readonly CarControlOutput[] {
  return Array.from({ length: carCount }, () => ({ throttle: 0, steer: 0 }));
}
