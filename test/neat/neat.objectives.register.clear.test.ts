import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

/** Tests for multi-objective registration and clearing logic. */
describe('NEAT Objectives Management', () => {
  describe('default fitness objective presence', () => {
    /** Constant fitness for deterministic baseline. */
    const fitness = (_: Network) => 1;
    /** Instance without multi-objective overrides. */
    const neat = new Neat(2, 1, fitness, { popsize: 3, seed: 444 });
    test('includes built-in fitness objective key', () => {
      // Arrange & Act: retrieve objective keys
      const keys = neat.getObjectiveKeys();
      // Assert: fitness key present
      expect(keys.includes('fitness')).toBe(true);
    });
  });
  describe('registering and clearing custom objectives', () => {
    /** Score proportional to connection count. */
    const fitness = (n: Network) => n.connections.length;
    /** Instance with multi-objective enabled. */
    const neat = new Neat(2, 1, fitness, {
      popsize: 4,
      seed: 445,
      multiObjective: { enabled: true },
    });
    test('registerObjective adds new key', () => {
      // Arrange: register custom objective
      (neat as any).registerObjective(
        'sparsity',
        'min',
        (g: any) => g.connections.length
      );
      // Act: retrieve keys including new objective
      const keys = neat.getObjectiveKeys();
      // Assert: custom key present
      expect(keys.includes('sparsity')).toBe(true);
    });
    test('clearObjectives removes custom objectives (keeps fitness)', () => {
      // Arrange: ensure a custom objective exists then clear
      (neat as any).registerObjective(
        'temp',
        'max',
        (g: any) => g.nodes.length
      );
      (neat as any).clearObjectives();
      // Act: get resulting keys
      const keys = neat.getObjectiveKeys();
      // Assert: only fitness remains
      expect(keys).toEqual(['fitness']);
    });
  });
});
