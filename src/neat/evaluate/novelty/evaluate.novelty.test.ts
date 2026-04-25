import { runNoveltyBlendAndArchive } from './evaluate.novelty';
import type {
  GenomeForEvaluation,
  NeatControllerForEval,
} from '../shared/evaluate.types';

type NoveltyDescriptorGenome = GenomeForEvaluation & {
  descriptorValue: number;
};

function createNoveltyGenome(input: {
  descriptorValue: number;
  score: number;
}): NoveltyDescriptorGenome {
  return {
    connections: [],
    descriptorValue: input.descriptorValue,
    score: input.score,
  };
}

function hasDescriptorValue(
  genome: GenomeForEvaluation,
): genome is NoveltyDescriptorGenome {
  return (
    'descriptorValue' in genome && typeof genome.descriptorValue === 'number'
  );
}

function createNoveltyController(): NeatControllerForEval {
  return {
    options: {
      novelty: {
        enabled: true,
        k: 1,
        blendFactor: 0.5,
        archiveAddThreshold: 0,
        descriptor: (genome) => [
          hasDescriptorValue(genome) ? genome.descriptorValue : 0,
        ],
      },
    },
    population: [
      createNoveltyGenome({ descriptorValue: 0, score: 10 }),
      createNoveltyGenome({ descriptorValue: 3, score: 20 }),
      createNoveltyGenome({ descriptorValue: 8, score: 30 }),
    ],
    fitness: async (genomeOrPopulation) => {
      void genomeOrPopulation;
      return 0;
    },
  };
}

describe('neat evaluate novelty chapter', () => {
  describe('runNoveltyBlendAndArchive', () => {
    describe('given deterministic descriptor distances with archive writes enabled', () => {
      it('records novelty scores, blends them into existing scores, and appends each descriptor to the archive', () => {
        // Arrange
        const evaluationController = createNoveltyController();

        // Act
        runNoveltyBlendAndArchive(
          evaluationController,
          evaluationController.options,
        );

        // Assert
        expect({
          scoredPopulation: evaluationController.population.map((genome) => ({
            score: genome.score,
            novelty: genome._novelty,
          })),
          archiveSize: evaluationController._noveltyArchive?.length,
        }).toEqual({
          scoredPopulation: [
            { score: 6.5, novelty: 3 },
            { score: 11.5, novelty: 3 },
            { score: 17.5, novelty: 5 },
          ],
          archiveSize: 3,
        });
      });
    });

    describe('given novelty is disabled', () => {
      it('returns early without mutating scores or novelty metadata', () => {
        // Arrange
        const evaluationController = createNoveltyController();
        evaluationController.options.novelty = {
          ...evaluationController.options.novelty,
          enabled: false,
        };

        // Act
        runNoveltyBlendAndArchive(
          evaluationController,
          evaluationController.options,
        );

        // Assert
        expect(
          evaluationController.population.map((genome) => ({
            score: genome.score,
            novelty: genome._novelty,
          })),
        ).toEqual([
          { score: 10, novelty: undefined },
          { score: 20, novelty: undefined },
          { score: 30, novelty: undefined },
        ]);
      });
    });

    describe('given a descriptor callback that throws', () => {
      it('falls back to empty descriptors and keeps novelty scoring resilient', () => {
        // Arrange
        const evaluationController = createNoveltyController();
        evaluationController.options.novelty = {
          ...evaluationController.options.novelty,
          descriptor: () => {
            throw new Error('descriptor failure');
          },
        };

        // Act
        runNoveltyBlendAndArchive(
          evaluationController,
          evaluationController.options,
        );

        // Assert
        expect(
          evaluationController.population.map((genome) => ({
            score: genome.score,
            novelty: genome._novelty,
          })),
        ).toEqual([
          { score: 5, novelty: 0 },
          { score: 10, novelty: 0 },
          { score: 15, novelty: 0 },
        ]);
      });
    });
  });
});
