import {
  _updateSpeciesStagnation,
  _sortSpeciesMembers,
} from '../../src/neat/neat.speciation';

describe('speciation - stagnation', () => {
  test('prunes species that exceed stagnation window', () => {
    // Arrange
    const stale: any = {
      id: 1,
      members: [{ score: 1 }],
      bestScore: 1, // matches top member so bestScore not improved
      lastImproved: 0, // far in past
    };
    const fresh: any = {
      id: 2,
      members: [{ score: 5 }],
      bestScore: 5, // matches member score avoids changing lastImproved
      lastImproved: 15, // within window (generation 20 - 15 =5 <=10)
    };
    const ctx: any = {
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
