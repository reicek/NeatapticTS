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

    describe('given falsy k and undefined blendFactor', () => {
      it('uses neighbor-count and blend-factor defaults without crashing', () => {
        // Arrange — k: 0 exercises the || default arm; blendFactor undefined exercises the ?? default arm.
        const evaluationController = createNoveltyController();
        evaluationController.options.novelty = {
          ...evaluationController.options.novelty,
          k: 0,
          blendFactor: undefined,
          archiveAddThreshold: 0,
        };

        // Act
        runNoveltyBlendAndArchive(
          evaluationController,
          evaluationController.options,
        );

        // Assert — novelty annotation is written for every genome (defaults resolved without error).
        expect(
          evaluationController.population.every(
            (genome) => typeof genome._novelty === 'number',
          ),
        ).toBe(true);
      });
    });

    describe('given descriptor returning null', () => {
      it('falls back to an empty descriptor vector for that genome', () => {
        // Arrange — returning null exercises the ?? [] fallback arm at line 171.
        const evaluationController = createNoveltyController();
        evaluationController.options.novelty = {
          ...evaluationController.options.novelty,
          descriptor: () => null as unknown as number[],
          archiveAddThreshold: 0,
        };

        // Act
        runNoveltyBlendAndArchive(
          evaluationController,
          evaluationController.options,
        );

        // Assert — all descriptors fell back to [], so all pairwise distances are 0, novelty is 0.
        expect(
          evaluationController.population.every(
            (genome) => genome._novelty === 0,
          ),
        ).toBe(true);
      });
    });

    describe('given a single-genome population', () => {
      it('assigns novelty 0 because there are no neighbors after excluding self-distance', () => {
        // Arrange — single genome means distanceRow is [0], slice(1, k+1) is [], exercises neighbors.length === 0 return path.
        const evaluationController = createNoveltyController();
        evaluationController.population = [
          createNoveltyGenome({ descriptorValue: 5, score: 10 }),
        ];

        // Act
        runNoveltyBlendAndArchive(
          evaluationController,
          evaluationController.options,
        );

        // Assert
        expect(evaluationController.population[0]?._novelty).toBe(0);
      });
    });

    describe('given a genome with a non-number score', () => {
      it('skips blending and leaves novelty annotated without touching score', () => {
        // Arrange — score: undefined exercises the typeof !== 'number' early-return guard in blendNoveltyIntoScore.
        const genomeWithoutScore: GenomeForEvaluation = {
          connections: [],
        };
        const evaluationController = createNoveltyController();
        evaluationController.population = [genomeWithoutScore];

        // Act
        runNoveltyBlendAndArchive(
          evaluationController,
          evaluationController.options,
        );

        // Assert — novelty is annotated but score is still undefined (blending skipped).
        expect({
          score: genomeWithoutScore.score,
          hasNovelty: typeof genomeWithoutScore._novelty === 'number',
        }).toEqual({ score: undefined, hasNovelty: true });
      });
    });

    describe('given no archiveAddThreshold configured', () => {
      it('skips archive admission because novelty never exceeds Infinity', () => {
        // Arrange — omitting archiveAddThreshold exercises the ?? Infinity arm (line 316) and the
        // shouldAdd=false path so !shouldAdd early-returns without appending (lines 318-319).
        const evaluationController = createNoveltyController();
        evaluationController.options.novelty = {
          enabled: true,
          k: 1,
          descriptor: (genome) => [
            hasDescriptorValue(genome) ? genome.descriptorValue : 0,
          ],
        };

        // Act
        runNoveltyBlendAndArchive(
          evaluationController,
          evaluationController.options,
        );

        // Assert — archive stays empty because novelty < Infinity for all genomes.
        expect(evaluationController._noveltyArchive?.length ?? 0).toBe(0);
      });
    });

    describe('given a pre-filled novelty archive at capacity', () => {
      it('does not append new descriptors when the archive is already at NOVELTY_ARCHIVE_CAP', () => {
        // Arrange — pre-fill to 200 entries to exercise the archive-full guard at line 322.
        const evaluationController = createNoveltyController();
        const archiveCapacity = 200;
        evaluationController._noveltyArchive = Array.from(
          { length: archiveCapacity },
          (_, archiveIndex) => ({ desc: [archiveIndex], novelty: archiveIndex }),
        );

        // Act — archiveAddThreshold: 0 ensures shouldAdd is true for every genome, but archive is full.
        runNoveltyBlendAndArchive(
          evaluationController,
          evaluationController.options,
        );

        // Assert — archive stays at exactly 200; no entries appended.
        expect(evaluationController._noveltyArchive.length).toBe(
          archiveCapacity,
        );
      });
    });
  });
});
