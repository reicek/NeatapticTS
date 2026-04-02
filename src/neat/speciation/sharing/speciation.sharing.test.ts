import {
  applyFitnessSharing,
  updateSpeciesStagnation,
} from './speciation.sharing.utils';
import type {
  GenomeDetailed,
  SpeciesLike,
} from '../../shared/neat.shared.types';

type FitnessSharingContext = {
  _species: SpeciesLike[];
  options: { sharingSigma?: number } & Record<string, unknown>;
  _compatibilityDistance: (
    leftGenome: GenomeDetailed,
    rightGenome: GenomeDetailed,
  ) => number;
};

type StagnationContext = {
  _species: SpeciesLike[];
  generation: number;
  options: { stagnationGenerations?: number } & Record<string, unknown>;
};

function createMember(genomeId: number, score: number): GenomeDetailed {
  return {
    _id: genomeId,
    nodes: [],
    connections: [],
    score,
  };
}

function createFitnessSharingContext(
  members: GenomeDetailed[],
  compatibilityDistanceValue: number,
): FitnessSharingContext {
  return {
    _species: [{ id: 1, members }],
    options: {},
    _compatibilityDistance: (
      leftGenome: GenomeDetailed,
      rightGenome: GenomeDetailed,
    ) => {
      void leftGenome;
      void rightGenome;
      return compatibilityDistanceValue;
    },
  };
}

function calculateMeanScore(members: GenomeDetailed[]): number {
  return (
    members.reduce(
      (scoreTotal, member) => scoreTotal + (member.score ?? 0),
      0,
    ) / members.length
  );
}

describe('neat speciation sharing chapter', () => {
  describe('applyFitnessSharing', () => {
    describe('given a species using uniform sharing', () => {
      it('divides each member score by the species size', () => {
        // Arrange
        const firstMember = createMember(1, 9);
        const secondMember = createMember(2, 3);
        const species: SpeciesLike = {
          id: 1,
          members: [firstMember, secondMember],
        };
        const speciationContext: FitnessSharingContext = {
          _species: [species],
          options: { sharingSigma: 0 },
          _compatibilityDistance: (
            leftGenome: GenomeDetailed,
            rightGenome: GenomeDetailed,
          ) => {
            void leftGenome;
            void rightGenome;
            return 0;
          },
        };

        // Act
        applyFitnessSharing(speciationContext, 0);

        // Assert
        expect(firstMember.score).toBe(9 / 2);
      });
    });

    describe('given a species using sigma-aware sharing', () => {
      it('reduces a member score when close neighbors contribute to the kernel sum', () => {
        // Arrange
        const firstMember = createMember(1, 10);
        const secondMember = createMember(2, 10);
        const species: SpeciesLike = {
          id: 1,
          members: [firstMember, secondMember],
        };
        const speciationContext: FitnessSharingContext = {
          _species: [species],
          options: { sharingSigma: 5 },
          _compatibilityDistance: (
            leftGenome: GenomeDetailed,
            rightGenome: GenomeDetailed,
          ) => {
            void leftGenome;
            void rightGenome;
            return 1;
          },
        };

        // Act
        applyFitnessSharing(speciationContext, 5);

        // Assert
        expect(firstMember.score ?? 0).toBeLessThan(10);
      });
    });

    describe('given the same crowded species under uniform and sigma-aware sharing', () => {
      it('keeps the sigma-aware mean adjusted score at or above the uniform baseline', () => {
        // Arrange
        const uniformMembers = [createMember(1, 10), createMember(2, 10)];
        const sigmaMembers = [createMember(3, 10), createMember(4, 10)];
        const uniformContext = createFitnessSharingContext(uniformMembers, 1);
        const sigmaContext = createFitnessSharingContext(sigmaMembers, 1);

        // Act
        applyFitnessSharing(uniformContext, 0);
        applyFitnessSharing(sigmaContext, 2);

        // Assert
        expect(calculateMeanScore(sigmaMembers)).toBeGreaterThanOrEqual(
          calculateMeanScore(uniformMembers),
        );
      });
    });
  });

  describe('updateSpeciesStagnation', () => {
    describe('given one stale species and one recently improved species', () => {
      it('prunes the species that exceeded the stagnation window', () => {
        // Arrange
        const staleSpecies: SpeciesLike = {
          id: 1,
          members: [createMember(1, 1)],
          bestScore: 1,
          lastImproved: 0,
        };
        const freshSpecies: SpeciesLike = {
          id: 2,
          members: [createMember(2, 5)],
          bestScore: 5,
          lastImproved: 15,
        };
        const speciationContext: StagnationContext = {
          _species: [staleSpecies, freshSpecies],
          generation: 20,
          options: { stagnationGenerations: 10 },
        };

        // Act
        updateSpeciesStagnation(speciationContext, 10, sortSpeciesMembers);

        // Assert
        expect(speciationContext._species).toHaveLength(1);
      });
    });
  });
});

function sortSpeciesMembers(species: SpeciesLike): void {
  species.members.sort(
    (leftMember, rightMember) =>
      ((rightMember as GenomeDetailed).score ?? 0) -
      ((leftMember as GenomeDetailed).score ?? 0),
  );
}
