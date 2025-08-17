import { _applyFitnessSharing } from '../../src/neat/neat.speciation';

describe('speciation - fitness sharing', () => {
  test('kernel sharing reduces fitness with close neighbors', () => {
    // Arrange
    const m1: any = { score: 10, nodes: [], connections: [], _id: 1 };
    const m2: any = { score: 10, nodes: [], connections: [], _id: 2 };
    const species = { members: [m1, m2] };
    const ctx: any = {
      _species: [species],
      options: { sharingSigma: 5 },
      _compatibilityDistance: () => 1, // within sigma
    };

    // Act
    _applyFitnessSharing.call(ctx);

    // Assert
    expect(m1.score).toBeLessThan(10);
  });
});
