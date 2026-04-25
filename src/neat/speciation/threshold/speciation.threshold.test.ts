import { adjustCompatibilityThreshold } from './speciation.threshold.utils';
import {
  DEFAULT_COMPAT_INTEGRAL,
  DEFAULT_TARGET_SPECIES,
} from '../shared/speciation.shared';
import type {
  ConnectionLike,
  GenomeDetailed,
  SpeciationHarnessContext,
  SpeciationOptions,
  SpeciesLastStats,
} from '../../shared/neat.shared.types';

type SpeciationPidOptions = SpeciationOptions & {
  compatibilityThreshold: number;
  compatAdjust: Required<NonNullable<SpeciationOptions['compatAdjust']>>;
};

type SpeciationPidContext = SpeciationHarnessContext<SpeciationPidOptions>;

function buildThresholdContext(
  population: GenomeDetailed[],
  optionsOverride: Partial<SpeciationPidContext['options']>,
  compatibilityIntegral: number,
): SpeciationPidContext {
  const defaultOptions: SpeciationPidContext['options'] = {
    speciation: true,
    targetSpecies: 1,
    compatibilityThreshold: 9.9,
    compatAdjust: {
      kp: 1,
      ki: 0.5,
      smoothingWindow: 1,
      minThreshold: 0.5,
      maxThreshold: 10,
      decay: 1,
    },
  };

  return {
    population,
    _species: [],
    _nextSpeciesId: 1,
    generation: 0,
    options: { ...defaultOptions, ...optionsOverride },
    _speciesCreated: new Map<number, number>(),
    _prevSpeciesMembers: new Map<number, Set<number>>(),
    _speciesLastStats: new Map<number, SpeciesLastStats>(),
    _speciesHistory: [],
    _compatIntegral: compatibilityIntegral,
    _getRNG: () => () => 0.5,
    _compatibilityDistance: (
      leftGenome: GenomeDetailed,
      rightGenome: GenomeDetailed,
    ) => {
      void leftGenome;
      void rightGenome;
      return 0;
    },
    _fallbackInnov: (connection: ConnectionLike) => {
      void connection;
      return 1;
    },
    _structuralEntropy: (genomeDetailed: GenomeDetailed) => {
      void genomeDetailed;
      return 0;
    },
  };
}

describe('neat speciation threshold chapter', () => {
  describe('adjustCompatibilityThreshold', () => {
    describe('given the public compatibility threshold is missing', () => {
      it('initializes the integral to the shared default and leaves the threshold unset', () => {
        // Arrange
        const population: GenomeDetailed[] = [
          { nodes: [], connections: [], _id: 1, score: 1 },
        ];
        const speciationContext = buildThresholdContext(
          population,
          {},
          undefined as unknown as number,
        );
        Reflect.set(speciationContext.options, 'compatibilityThreshold', undefined);

        // Act
        adjustCompatibilityThreshold(
          speciationContext,
          speciationContext.options,
          speciationContext.options.compatAdjust,
          speciationContext.options.compatAdjust.minThreshold,
          speciationContext.options.compatAdjust.maxThreshold,
        );

        // Assert
        expect({
          compatibilityThreshold:
            speciationContext.options.compatibilityThreshold ?? null,
          compatibilityIntegral: speciationContext._compatIntegral,
        }).toEqual({
          compatibilityThreshold: null,
          compatibilityIntegral: 0,
        });
      });
    });

    describe('given the PID update stays inside the configured bounds', () => {
      it('keeps the computed threshold without resetting the integral', () => {
        // Arrange
        const population: GenomeDetailed[] = [
          { nodes: [], connections: [], _id: 1, score: 1 },
        ];
        const speciationContext = buildThresholdContext(
          population,
          {
            targetSpecies: 1,
            compatibilityThreshold: 6,
          },
          0,
        );
        speciationContext._species = [{ id: 1, members: [population[0]] } as never];

        // Act
        adjustCompatibilityThreshold(
          speciationContext,
          speciationContext.options,
          speciationContext.options.compatAdjust,
          speciationContext.options.compatAdjust.minThreshold,
          speciationContext.options.compatAdjust.maxThreshold,
        );

        // Assert
        expect({
          compatibilityIntegral: speciationContext._compatIntegral,
          compatibilityThreshold: speciationContext.options.compatibilityThreshold,
        }).toEqual({
          compatibilityIntegral: 0,
          compatibilityThreshold: 6,
        });
      });
    });

    describe('given the PID update relies on the shared target, gain, and integral defaults', () => {
      it('keeps the threshold unchanged while hydrating the missing defaults', () => {
        // Arrange
        const population: GenomeDetailed[] = [
          { nodes: [], connections: [], _id: 1, score: 1 },
        ];
        const speciationContext = buildThresholdContext(
          population,
          {
            compatAdjust: {
              maxThreshold: 10,
              minThreshold: 0.5,
            } as SpeciationPidContext['options']['compatAdjust'],
            compatibilityThreshold: 6,
            targetSpecies: undefined,
          },
          DEFAULT_COMPAT_INTEGRAL,
        );
        speciationContext._species = Array.from(
          { length: DEFAULT_TARGET_SPECIES },
          (_, speciesIndex) => ({
            id: speciesIndex + 1,
            members: [population[0]],
          }),
        ) as never;
        Reflect.set(speciationContext, '_compatIntegral', undefined);

        // Act
        adjustCompatibilityThreshold(
          speciationContext,
          speciationContext.options,
          speciationContext.options.compatAdjust,
          0.5,
          10,
        );

        // Assert
        expect({
          compatibilityIntegral: speciationContext._compatIntegral,
          compatibilityThreshold: speciationContext.options.compatibilityThreshold,
        }).toEqual({
          compatibilityIntegral: DEFAULT_COMPAT_INTEGRAL,
          compatibilityThreshold: 6,
        });
      });
    });

    describe('given a threshold update that falls below the configured minimum', () => {
      it('resets the compatibility integral after low clipping', () => {
        // Arrange
        const population: GenomeDetailed[] = [
          { nodes: [], connections: [], _id: 1, score: 1 },
        ];
        const speciationContext = buildThresholdContext(
          population,
          {
            targetSpecies: 5,
            compatibilityThreshold: 0.6,
          },
          100,
        );

        // Act
        adjustCompatibilityThreshold(
          speciationContext,
          speciationContext.options,
          speciationContext.options.compatAdjust,
          speciationContext.options.compatAdjust.minThreshold,
          speciationContext.options.compatAdjust.maxThreshold,
        );

        // Assert
        expect(speciationContext._compatIntegral).toBe(0);
      });

      it('clamps the compatibility threshold to the configured minimum', () => {
        // Arrange
        const population: GenomeDetailed[] = [
          { nodes: [], connections: [], _id: 1, score: 1 },
        ];
        const speciationContext = buildThresholdContext(
          population,
          {
            targetSpecies: 5,
            compatibilityThreshold: 0.6,
          },
          100,
        );

        // Act
        adjustCompatibilityThreshold(
          speciationContext,
          speciationContext.options,
          speciationContext.options.compatAdjust,
          speciationContext.options.compatAdjust.minThreshold,
          speciationContext.options.compatAdjust.maxThreshold,
        );

        // Assert
        expect(speciationContext.options.compatibilityThreshold).toBe(0.5);
      });
    });

    describe('given a threshold update that rises above the configured maximum', () => {
      it('resets the compatibility integral after high clipping', () => {
        // Arrange
        const population: GenomeDetailed[] = [
          { nodes: [], connections: [], _id: 1, score: 1 },
          { nodes: [], connections: [], _id: 2, score: 1 },
        ];
        const speciationContext = buildThresholdContext(population, {}, -100);

        // Act
        adjustCompatibilityThreshold(
          speciationContext,
          speciationContext.options,
          speciationContext.options.compatAdjust,
          speciationContext.options.compatAdjust.minThreshold,
          speciationContext.options.compatAdjust.maxThreshold,
        );

        // Assert
        expect(speciationContext._compatIntegral).toBe(0);
      });

      it('clamps the compatibility threshold to the configured maximum', () => {
        // Arrange
        const population: GenomeDetailed[] = [
          { nodes: [], connections: [], _id: 1, score: 1 },
          { nodes: [], connections: [], _id: 2, score: 1 },
        ];
        const speciationContext = buildThresholdContext(population, {}, -100);

        // Act
        adjustCompatibilityThreshold(
          speciationContext,
          speciationContext.options,
          speciationContext.options.compatAdjust,
          speciationContext.options.compatAdjust.minThreshold,
          speciationContext.options.compatAdjust.maxThreshold,
        );

        // Assert
        expect(speciationContext.options.compatibilityThreshold).toBe(10);
      });
    });
  });
});
