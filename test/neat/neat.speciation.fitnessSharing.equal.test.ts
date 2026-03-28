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
  test('equal sharing divides fitness by species size', () => {
    // Arrange
    const memberOne: GenomeDetailed = {
      score: 9,
      nodes: [],
      connections: [],
      _id: 1,
    };
    const memberTwo: GenomeDetailed = {
      score: 3,
      nodes: [],
      connections: [],
      _id: 2,
    };
    const species: SpeciesLike = { id: 1, members: [memberOne, memberTwo] };
    const ctx: FitnessSharingContext = {
      _species: [species],
      options: { sharingSigma: 0 },
      _compatibilityDistance: (
        genomeA: GenomeDetailed,
        genomeB: GenomeDetailed,
      ) => {
        void genomeA;
        void genomeB;
        return 0;
      },
    };

    // Act
    _applyFitnessSharing.call(ctx);

    // Assert
    expect(memberOne.score).toBe(9 / 2);
  });
});
