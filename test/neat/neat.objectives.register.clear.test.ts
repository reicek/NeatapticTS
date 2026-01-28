import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

/** Tests for multi-objective registration and clearing logic. */
describe('NEAT Objectives Management', () => {
  describe('default fitness objective presence', () => {
    /** Constant fitness for deterministic baseline. */
    const fitness = () => 1;
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
    const neatWithObjectives = neat as unknown as {
      registerObjective: (
        key: string,
        direction: 'min' | 'max',
        accessor: (network: Network) => number,
      ) => void;
      clearObjectives: () => void;
    };
    test('registerObjective adds new key', () => {
      // Arrange: register custom objective
      neatWithObjectives.registerObjective(
        'sparsity',
        'min',
        (network) => network.connections.length,
      );
      // Act: retrieve keys including new objective
      const keys = neat.getObjectiveKeys();
      // Assert: custom key present
      expect(keys.includes('sparsity')).toBe(true);
    });
    test('clearObjectives removes custom objectives (keeps fitness)', () => {
      // Arrange: ensure a custom objective exists then clear
      neatWithObjectives.registerObjective(
        'temp',
        'max',
        (network) => network.nodes.length,
      );
      neatWithObjectives.clearObjectives();
      // Act: get resulting keys
      const keys = neat.getObjectiveKeys();
      // Assert: only fitness remains
      expect(keys).toEqual(['fitness']);
    });
  });
});
