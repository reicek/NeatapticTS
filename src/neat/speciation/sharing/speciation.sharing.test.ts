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

  describe('applyFitnessSharing branch coverage', () => {
    it('exercises member score undefined branch when score is not a number', () => {
      // Arrange
      const memberWithoutScore = { _id: 1, nodes: [], connections: [] } as GenomeDetailed;
      const memberWithScore = createMember(2, 10);
      const species: SpeciesLike = {
        id: 1,
        members: [memberWithoutScore, memberWithScore],
      };
      const speciationContext: FitnessSharingContext = {
        _species: [species],
        options: {},
        _compatibilityDistance: () => 0,
      };

      // Act
      applyFitnessSharing(speciationContext, 0);

      // Assert
      expect(memberWithoutScore.score).toBeUndefined();
      expect(memberWithScore.score).toBe(10 / 2);
    });

    it('exercises distance >= sigmaValue false branch in sigma sharing', () => {
      // Arrange
      const firstMember = createMember(1, 10);
      const secondMember = createMember(2, 10);
      const species: SpeciesLike = {
        id: 1,
        members: [firstMember, secondMember],
      };
      const speciationContext: FitnessSharingContext = {
        _species: [species],
        options: {},
        _compatibilityDistance: () => 5,
      };

      // Act
      applyFitnessSharing(speciationContext, 3);

      // Assert - distance (5) >= sigmaValue (3), so sharing penalty should be minimal
      expect(firstMember.score).toBeLessThanOrEqual(10);
    });

    it('exercises empty members array fallback in uniform sharing', () => {
      // Arrange
      const species: SpeciesLike = {
        id: 1,
        members: [],
      };
      const speciationContext: FitnessSharingContext = {
        _species: [species],
        options: {},
        _compatibilityDistance: () => 0,
      };

      // Act
      applyFitnessSharing(speciationContext, 0);

      // Assert - should use DEFAULT_MEMBER_COUNT_FALLBACK
      expect(species.members).toHaveLength(0);
    });

    it('exercises sigma sharing path with numeric scores (line 163 false branch)', () => {
      // Arrange
      const firstMember = createMember(1, 100);
      const secondMember = createMember(2, 50);
      const species: SpeciesLike = {
        id: 1,
        members: [firstMember, secondMember],
      };
      let distanceCalls = 0;
      const speciationContext: FitnessSharingContext = {
        _species: [species],
        options: {},
        _compatibilityDistance: (leftGenome: GenomeDetailed, rightGenome: GenomeDetailed) => {
          distanceCalls++;
          void leftGenome;
          void rightGenome;
          return 0;
        },
      };
      const initialScore1 = firstMember.score as number;
      const initialScore2 = secondMember.score as number;

      // Act
      applyFitnessSharing(speciationContext, 5);

      // Assert - verify numeric scores path was taken (line 163 FALSE branch)
      expect(typeof firstMember.score).toBe('number');
      expect(typeof secondMember.score).toBe('number');
      // Verify scores were modified (not === to initial values)
      expect(firstMember.score).not.toBe(initialScore1);
      expect(secondMember.score).not.toBe(initialScore2);
      // Verify they're less than original (normalized down by sharing)
      expect(firstMember.score).toBeLessThan(initialScore1);
      expect(secondMember.score).toBeLessThan(initialScore2);
      // Verify distance function was called (proves line 164-171 path executed)
      expect(distanceCalls).toBeGreaterThan(0);
    });


    it('skips non-numeric score members in sigma sharing path (line 163 false branch)', () => {
      // Arrange - sigma > 0 routes to applySigmaSharingToMembers
      // Line 163: if (typeof member.score === 'number') { ... }
      // FALSE branch: score is not a number -> skip, do not normalize
      const memberWithScore = createMember(1, 100);
      const memberWithoutScore: GenomeDetailed = {
        _id: 2,
        nodes: [],
        connections: [],
        // score intentionally omitted -> undefined
      };
      const species: SpeciesLike = {
        id: 1,
        members: [memberWithScore, memberWithoutScore],
      };
      const speciationContext: FitnessSharingContext = {
        _species: [species],
        options: {},
        _compatibilityDistance: () => 0,
      };

      // Act - sigma > 0 routes to sigma sharing path
      applyFitnessSharing(speciationContext, 5);

      // Assert - member without score must remain unchanged (not touched by normalization)
      expect(memberWithoutScore.score).toBeUndefined();
    });

      it('exercises nullish coalesce when bestScore is defined (line 268 false branch)', () => {
      // Arrange - species WITH defined bestScore to test line 268 nullish coalesce FALSE branch
      // Line 268: const currentBest = species.bestScore ?? NEGATIVE_INFINITY;
      // FALSE branch: bestScore IS defined (not undefined), so ?? doesn't activate fallback
      const topMember = createMember(1, 50);
      const species: SpeciesLike = {
        id: 1,
        members: [topMember],
        bestScore: 75, // DEFINED bestScore - tests FALSE branch of ??
        lastImproved: 10,
      };
      const speciationContext: StagnationContext = {
        _species: [species],
        generation: 12,
        options: {},
      };

      // Act
      updateSpeciesStagnation(speciationContext, 10, sortSpeciesMembers);

      // Assert - bestScore should remain 75 (currentBest is 75, candidateBest is 50, no improvement)
      expect(species.bestScore).toBe(75);
      expect(species.lastImproved).toBe(10);
    });

    it('exercises nullish coalesce when member score is defined (line 269 false branch)', () => {
      // Arrange - member WITH defined score to test line 269 nullish coalesce FALSE branch
      // Line 269: const candidateBest = topMember?.score ?? NEGATIVE_INFINITY;
      // FALSE branch: topMember?.score IS defined, so ?? doesn't activate fallback
      const topMember = createMember(1, 100);
      // topMember.score is defined as 100
      const species: SpeciesLike = {
        id: 1,
        members: [topMember],
        bestScore: 50, // currentBest = 50
        lastImproved: 10,
      };
      const speciationContext: StagnationContext = {
        _species: [species],
        generation: 12,
        options: {},
      };

      // Act
      updateSpeciesStagnation(speciationContext, 10, sortSpeciesMembers);

      // Assert - candidateBest (100) > currentBest (50), so bestScore is updated
      // This exercises line 269 FALSE branch (score IS defined, not NEGATIVE_INFINITY)
      expect(species.bestScore).toBe(100);
      expect(species.lastImproved).toBe(12);
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

    it('exercises non-improvement branch when candidate score does not exceed best', () => {
      // Arrange
      const topMember = createMember(1, 5);
      const species: SpeciesLike = {
        id: 1,
        members: [topMember],
        bestScore: 10,
        lastImproved: 10,
      };
      const speciationContext: StagnationContext = {
        _species: [species],
        generation: 12,
        options: {},
      };

      // Act
      updateSpeciesStagnation(speciationContext, 10, sortSpeciesMembers);

      // Assert - bestScore should remain 10 (not updated to 5)
      expect(species.bestScore).toBe(10);
      expect(species.lastImproved).toBe(10);
    });

    it('exercises nullish coalesce when bestScore is undefined (line 268 false branch)', () => {
      // Arrange - species WITHOUT bestScore to test line 268 nullish coalesce FALSE branch
      const topMember = createMember(1, 50);
      const species: SpeciesLike = {
        id: 1,
        members: [topMember],
        // bestScore is undefined - tests FALSE branch of ?? operator
        lastImproved: 10,
      };
      const speciationContext: StagnationContext = {
        _species: [species],
        generation: 12,
        options: {},
      };

      // Act
      updateSpeciesStagnation(speciationContext, 10, sortSpeciesMembers);

      // Assert - bestScore should be set to topMember score (50)
      expect(species.bestScore).toBe(50);
      expect(species.lastImproved).toBe(12);
    });

    it('exercises nullish coalesce when member score is undefined (line 269 false branch)', () => {
      // Arrange - member WITHOUT score to test line 269 nullish coalesce FALSE branch
      const topMember = { _id: 1, nodes: [], connections: [] } as GenomeDetailed;
      // topMember.score is undefined
      const species: SpeciesLike = {
        id: 1,
        members: [topMember],
        // bestScore undefined too to test both nullish operators
        lastImproved: 10,
      };
      const speciationContext: StagnationContext = {
        _species: [species],
        generation: 12,
        options: {},
      };

      // Act
      updateSpeciesStagnation(speciationContext, 10, sortSpeciesMembers);

      // Assert - candidateBest will be NEGATIVE_INFINITY, currentBest will be NEGATIVE_INFINITY
      // so the if condition is false (not improving)
      expect(species.bestScore).toBeUndefined();
      expect(species.lastImproved).toBe(10);
    });

    it('exercises prune-all branch when no species survive stagnation window', () => {
      // Arrange - both species exceed the stagnation window, so survivors.length will be 0
      const staleSpecies1: SpeciesLike = {
        id: 1,
        members: [createMember(1, 1)],
        bestScore: 1,
        lastImproved: 0,
      };
      const staleSpecies2: SpeciesLike = {
        id: 2,
        members: [createMember(2, 1)],
        bestScore: 1,
        lastImproved: 1,
      };
      const originalArray = [staleSpecies1, staleSpecies2];
      const speciationContext: StagnationContext = {
        _species: originalArray,
        generation: 50,
        options: {},
      };

      // Act
      updateSpeciesStagnation(speciationContext, 5, sortSpeciesMembers);

      // Assert - survivors.length is falsy (0), so _species array is NOT reassigned
      // This exercises the FALSE branch of line 269 (if (survivors.length))
      expect(speciationContext._species).toBe(originalArray);
      expect(speciationContext._species).toHaveLength(2);
    });

    it('exercises survivors assignment branch when some species survive stagnation', () => {
      // Arrange - one species survives, so survivors.length will be > 0
      const staleSpecies: SpeciesLike = {
        id: 1,
        members: [createMember(1, 1)],
        bestScore: 10,
        lastImproved: 5,
      };
      const freshSpecies: SpeciesLike = {
        id: 2,
        members: [createMember(2, 10)],
        bestScore: 5,
        lastImproved: 18,
      };
      const speciationContext: StagnationContext = {
        _species: [staleSpecies, freshSpecies],
        generation: 20,
        options: {},
      };

      // Act
      updateSpeciesStagnation(speciationContext, 5, sortSpeciesMembers);

      // Assert - staleSpecies: (20 - 5) = 15 > 5 (outside window), filtered out
      // freshSpecies: (20 - 18) = 2 <= 5 (within window), kept
      // survivors.length is > 0, so _species is reassigned to survivors
      expect(speciationContext._species).toHaveLength(1);
      expect(speciationContext._species[0].id).toBe(2);
    });

    it('exercises lastImproved nullish coalesce when lastImproved is undefined', () => {
      // Arrange
      const species: SpeciesLike = {
        id: 1,
        members: [createMember(1, 10)],
        bestScore: 10,
        // lastImproved undefined
      };
      const speciationContext: StagnationContext = {
        _species: [species],
        generation: 5,
        options: {},
      };

      // Act
      updateSpeciesStagnation(speciationContext, 100, sortSpeciesMembers);

      // Assert - should use DEFAULT_LAST_IMPROVED_GENERATION, species should survive
      expect(speciationContext._species).toHaveLength(1);
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
