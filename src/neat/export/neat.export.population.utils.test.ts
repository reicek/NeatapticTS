import { NeatExportPopulationValidationError } from './neat.export.errors';
import {
  hydrateGenomeControllerMeta,
  serializeGenomeCheckpoint,
  splitSerializedGenomeCheckpoint,
} from './neat.export.population.utils';
import type {
  GenomeControllerCarrier,
  GenomeControllerMetaJSON,
  GenomeJSON,
} from './neat.export.types';

type MutableGenome = GenomeControllerCarrier & {
  _compatInnovationMode?: 'require-explicit' | 'allow-fallback';
  _crowdingDistance?: number;
  _depth?: number;
  _frontRank?: number;
  _id?: number;
  _moCrowd?: number;
  _moRank?: number;
  _novelty?: number;
  _parents?: number[];
  _reenableAttempts?: number;
  _reenableProb?: number;
  _reenableSuccess?: number;
  _sharedFitness?: number;
  _structuralEntropy?: number;
  score?: number;
};

function createHydrationHarness(): {
  genome: MutableGenome;
  getReceivedRngState: () => number | undefined;
} {
  let receivedRngState: number | undefined;

  return {
    genome: {
      setRNGState(state: number) {
        receivedRngState = state;
      },
      toJSON() {
        return {};
      },
    },
    getReceivedRngState() {
      return receivedRngState;
    },
  };
}

describe('neat export population utility chapter', () => {
  describe('serializeGenomeCheckpoint()', () => {
    describe('when the genome has no controller-owned metadata', () => {
      it('returns the raw network payload unchanged', () => {
        // Arrange
        const networkPayload: GenomeJSON = { marker: 'raw-network' };
        const genome: GenomeControllerCarrier = {
          toJSON() {
            return networkPayload;
          },
        };

        // Act
        const serializedGenome = serializeGenomeCheckpoint(genome);

        // Assert
        expect(serializedGenome).toBe(networkPayload);
      });
    });

    describe('when the genome carries controller-owned export metadata', () => {
      it('adds the reserved controllerMeta pocket with every supported field', () => {
        // Arrange
        const networkPayload: GenomeJSON = {
          connections: [{ innovation: 1 }],
          marker: 'raw-network',
        };
        const genome: MutableGenome = {
          _compatInnovationMode: 'allow-fallback',
          _crowdingDistance: 2.5,
          _depth: 9,
          _frontRank: 3,
          _id: 91,
          _moCrowd: 6.5,
          _moRank: 5,
          _novelty: 12.5,
          _parents: [7, 8],
          _reenableAttempts: 11,
          _reenableProb: 0.4,
          _reenableSuccess: 10,
          _sharedFitness: 1.5,
          _structuralEntropy: 4.5,
          getRNGState() {
            return 101;
          },
          score: 7,
          toJSON() {
            return networkPayload;
          },
        };

        // Act
        const serializedGenome = serializeGenomeCheckpoint(genome, networkPayload);

        // Assert
        expect(serializedGenome).toEqual({
          connections: [{ innovation: 1 }],
          controllerMeta: {
            compatInnovationMode: 'allow-fallback',
            crowdingDistance: 2.5,
            depth: 9,
            frontRank: 3,
            genomeId: 91,
            multiObjectiveCrowding: 6.5,
            multiObjectiveRank: 5,
            networkRngState: 101,
            novelty: 12.5,
            parents: [7, 8],
            reenableAttempts: 11,
            reenableProb: 0.4,
            reenableSuccess: 10,
            score: 7,
            sharedFitness: 1.5,
            structuralEntropy: 4.5,
          },
          marker: 'raw-network',
        });
      });
    });
  });

  describe('splitSerializedGenomeCheckpoint()', () => {
    describe('when controllerMeta is a plain object', () => {
      it('returns the reserved metadata separately from the raw payload', () => {
        // Arrange
        const serializedGenome: GenomeJSON = {
          controllerMeta: { genomeId: 4, novelty: 0.75 },
          marker: 'raw-network',
        };

        // Act
        const splitCheckpoint = splitSerializedGenomeCheckpoint(serializedGenome);

        // Assert
        expect(splitCheckpoint).toEqual({
          controllerMeta: { genomeId: 4, novelty: 0.75 },
          networkPayload: { marker: 'raw-network' },
        });
      });
    });

    describe('when controllerMeta is not a plain object', () => {
      it('drops the reserved metadata pocket and keeps only the network payload', () => {
        // Arrange
        const serializedGenome: GenomeJSON = {
          controllerMeta: [] as unknown as GenomeControllerMetaJSON,
          marker: 'raw-network',
        };

        // Act
        const splitCheckpoint = splitSerializedGenomeCheckpoint(serializedGenome);

        // Assert
        expect(splitCheckpoint).toEqual({
          controllerMeta: undefined,
          networkPayload: { marker: 'raw-network' },
        });
      });
    });
  });

  describe('hydrateGenomeControllerMeta()', () => {
    describe('when the checkpoint carries full controller metadata with a unique genome id', () => {
      it('restores every controller-owned field and advances the id floor above the imported id', () => {
        // Arrange
        const hydrationHarness = createHydrationHarness();
        const seenGenomeIds = new Set<number>();

        // Act
        const nextAssignedGenomeId = hydrateGenomeControllerMeta(
          hydrationHarness.genome,
          {
            compatInnovationMode: 'allow-fallback',
            crowdingDistance: 2.5,
            depth: 8,
            frontRank: 3,
            genomeId: 41,
            multiObjectiveCrowding: 7.5,
            multiObjectiveRank: 6,
            networkRngState: 5,
            novelty: 11.5,
            parents: [7, 'skip', 8] as unknown as number[],
            reenableAttempts: 10,
            reenableProb: 0.2,
            reenableSuccess: 9,
            score: 4,
            sharedFitness: 1.5,
            structuralEntropy: 4.5,
          },
          seenGenomeIds,
          15,
        );

        // Assert
        expect({
          compatInnovationMode: hydrationHarness.genome._compatInnovationMode,
          crowdingDistance: hydrationHarness.genome._crowdingDistance,
          depth: hydrationHarness.genome._depth,
          frontRank: hydrationHarness.genome._frontRank,
          genomeId: hydrationHarness.genome._id,
          multiObjectiveCrowding: hydrationHarness.genome._moCrowd,
          multiObjectiveRank: hydrationHarness.genome._moRank,
          nextAssignedGenomeId,
          novelty: hydrationHarness.genome._novelty,
          parents: hydrationHarness.genome._parents,
          receivedRngState: hydrationHarness.getReceivedRngState(),
          reenableAttempts: hydrationHarness.genome._reenableAttempts,
          reenableProb: hydrationHarness.genome._reenableProb,
          reenableSuccess: hydrationHarness.genome._reenableSuccess,
          score: hydrationHarness.genome.score,
          seenGenomeIds: [...seenGenomeIds],
          sharedFitness: hydrationHarness.genome._sharedFitness,
          structuralEntropy: hydrationHarness.genome._structuralEntropy,
        }).toEqual({
          compatInnovationMode: 'allow-fallback',
          crowdingDistance: 2.5,
          depth: 8,
          frontRank: 3,
          genomeId: 41,
          multiObjectiveCrowding: 7.5,
          multiObjectiveRank: 6,
          nextAssignedGenomeId: 42,
          novelty: 11.5,
          parents: [7, 8],
          receivedRngState: 5,
          reenableAttempts: 10,
          reenableProb: 0.2,
          reenableSuccess: 9,
          score: 4,
          seenGenomeIds: [41],
          sharedFitness: 1.5,
          structuralEntropy: 4.5,
        });
      });
    });

    describe('when the checkpoint reuses an existing genome id', () => {
      it('throws the population validation error', () => {
        // Arrange
        const hydrationHarness = createHydrationHarness();
        const seenGenomeIds = new Set<number>([6]);

        // Act
        const hydrateDuplicateGenomeId = () =>
          hydrateGenomeControllerMeta(
            hydrationHarness.genome,
            { genomeId: 6 },
            seenGenomeIds,
            20,
          );

        // Assert
        expect(hydrateDuplicateGenomeId).toThrow(
          NeatExportPopulationValidationError,
        );
      });
    });

    describe('when the checkpoint omits an explicit genome id', () => {
      it('assigns the next fallback genome id and increments the floor by one', () => {
        // Arrange
        const hydrationHarness = createHydrationHarness();
        const seenGenomeIds = new Set<number>();

        // Act
        const nextAssignedGenomeId = hydrateGenomeControllerMeta(
          hydrationHarness.genome,
          { score: 2 },
          seenGenomeIds,
          12,
        );

        // Assert
        expect({
          genomeId: hydrationHarness.genome._id,
          nextAssignedGenomeId,
          score: hydrationHarness.genome.score,
          seenGenomeIds: [...seenGenomeIds],
        }).toEqual({
          genomeId: 12,
          nextAssignedGenomeId: 13,
          score: 2,
          seenGenomeIds: [12],
        });
      });
    });
  });
});