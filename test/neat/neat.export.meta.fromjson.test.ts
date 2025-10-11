import Neat from '../../src/neat';
import Network from '../../src/architecture/network';
import type { NeatMetaJSON } from '../../src/neat/neat.export';

/** Additional tests for meta-only serialization (toJSON/fromJSON). */
describe('NEAT Meta Serialization', () => {
  describe('fromJSON restores generation & innovations', () => {
    /** Fitness proportional to node count for deterministic meta change. */
    const fitness = (n: Network) => n.nodes.length;
    const original = new Neat(2, 1, fitness, { popsize: 5, seed: 802 });
    let meta: NeatMetaJSON;
    beforeAll(async () => {
      await original.evaluate();
      await original.evolve();
      // Arrange: capture meta-only JSON (without population genomes)
      meta = original.toJSON();
    });
    test('fromJSON sets generation field', () => {
      // Act: rehydrate via static fromJSON
      const restored = Neat.fromJSON(meta, fitness);
      // Assert: generation matches exported meta
      expect(restored.generation).toBe(meta.generation);
    });
  });
});
