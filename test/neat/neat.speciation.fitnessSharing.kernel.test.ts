import { _applyFitnessSharing } from '../../src/neat/speciation/speciation';
import type {
  GenomeDetailed,
  SpeciesLike,
} from '../../src/neat/shared/neat.shared.types';

type FitnessSharingContext = {
  _species: SpeciesLike[];
  options: { sharingSigma?: number } & Record<string, unknown>;
  _compatibilityDistance: (
    genomeA: GenomeDetailed,
    genomeB: GenomeDetailed,
  ) => number;
};

describe('speciation - fitness sharing', () => {
  test('kernel sharing reduces fitness with close neighbors', () => {
    // Arrange
    const memberOne: GenomeDetailed = {
      score: 10,
      nodes: [],
      connections: [],
      _id: 1,
    };
    const memberTwo: GenomeDetailed = {
      score: 10,
      nodes: [],
      connections: [],
      _id: 2,
    };
    const species: SpeciesLike = { id: 1, members: [memberOne, memberTwo] };
    const ctx: FitnessSharingContext = {
      _species: [species],
      options: { sharingSigma: 5 },
      _compatibilityDistance: (
        genomeA: GenomeDetailed,
        genomeB: GenomeDetailed,
      ) => {
        void genomeA;
        void genomeB;
        return 1;
      },
    };

    // Act
    _applyFitnessSharing.call(ctx);

    // Assert
    expect(memberOne.score).toBeLessThan(10);
  });
});
