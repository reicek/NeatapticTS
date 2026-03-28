import {
  _updateSpeciesStagnation,
  _sortSpeciesMembers,
} from '../../src/neat/speciation/speciation';
import type {
  GenomeDetailed,
  SpeciesLike,
} from '../../src/neat/shared/neat.shared.types';

type StagnationContext = {
  _species: SpeciesLike[];
  generation: number;
  options: { stagnationGenerations?: number } & Record<string, unknown>;
  _sortSpeciesMembers: typeof _sortSpeciesMembers;
};

describe('speciation - stagnation', () => {
  test('prunes species that exceed stagnation window', () => {
    // Arrange
    const createMember = (id: number, score: number): GenomeDetailed => ({
      _id: id,
      nodes: [],
      connections: [],
      score,
    });
    const stale: SpeciesLike = {
      id: 1,
      members: [createMember(1, 1)],
      bestScore: 1,
      lastImproved: 0,
    };
    const fresh: SpeciesLike = {
      id: 2,
      members: [createMember(2, 5)],
      bestScore: 5,
      lastImproved: 15,
    };
    const ctx: StagnationContext = {
      _species: [stale, fresh],
      generation: 20,
      options: { stagnationGenerations: 10 },
      _sortSpeciesMembers,
    };

    // Act
    _updateSpeciesStagnation.call(ctx);

    // Assert
    expect(ctx._species.length).toBe(1);
  });
});
