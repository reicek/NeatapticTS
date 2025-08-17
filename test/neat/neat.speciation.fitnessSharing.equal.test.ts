import { _applyFitnessSharing } from '../../src/neat/neat.speciation';

describe('speciation - fitness sharing', () => {
  test('equal sharing divides fitness by species size', () => {
    // Arrange
    const m1: any = { score: 9, nodes: [], connections: [], _id: 1 };
    const m2: any = { score: 3, nodes: [], connections: [], _id: 2 };
    const species = { members: [m1, m2] };
    const ctx: any = {
      _species: [species],
      options: { sharingSigma: 0 },
    };

    // Act
    _applyFitnessSharing.call(ctx);

    // Assert
    expect(m1.score).toBe(9 / 2);
  });
});
