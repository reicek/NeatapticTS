/**
 * Tests for Neat.spawnFromParent and Neat.addGenome behaviors.
 * These tests follow the AAA pattern and use single-expect assertions as required.
 */
import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

/**
 * Top-level scenario grouping for spawn/add helpers of Neat.
 */
describe('Neat spawnFromParent and addGenome helpers', () => {
  /**
   * Factory to create a fresh Neat instance for each scenario.
   * We keep the fitness trivial since we only inspect bookkeeping side-effects.
   */
  const makeNeat = () => new Neat(3, 2, () => 0, { popsize: 6 });

  describe('spawnFromParent(parent, mutateCount)', () => {
    // Create a fresh neat and pick a parent from its initial pool
    const neat = makeNeat();
    const parent = neat.population[0];

    /**
     * Ensure spawnFromParent assigns a new unique genome id (distinct from parent).
     */
    it('should assign a new id distinct from parent', () => {
      // Arrange: have neat and parent defined above
      // Act: spawn a child
      const child = (neat as any).spawnFromParent(parent, 1);
      // Assert: child id must not equal parent id
      expect((child as any)._id).not.toBe((parent as any)._id);
    });

    /**
     * Ensure spawnFromParent records the parent id in _parents array.
     */
    it('should set parent id in _parents', () => {
      // Arrange
      const child = (neat as any).spawnFromParent(parent, 1);
      // Act is same as spawn
      // Assert: the first parent id equals parent's id
      expect((child as any)._parents).toEqual([(parent as any)._id]);
    });

    /**
     * Ensure the child's depth is parent's depth + 1 when lineage is enabled.
     */
    it('should set depth equal to parent.depth + 1', () => {
      // Arrange
      const baseDepth = (parent as any)._depth ?? 0;
      // Act
      const child = (neat as any).spawnFromParent(parent, 1);
      // Assert: child's depth increments parent's depth
      expect((child as any)._depth).toBe(baseDepth + 1);
    });

    /**
     * Ensure structural invariants are preserved (child has at least one connection).
     */
    it('should ensure the spawned child has at least one connection', () => {
      // Arrange/Act
      const child = (neat as any).spawnFromParent(parent, 1);
      // Assert: connections array length is greater than zero
      expect(child.connections.length).toBeGreaterThan(0);
    });
  });

  describe('addGenome(genome, parents?)', () => {
    // Create a fresh Neat for the addGenome scenarios
    const neat = makeNeat();
    const parent = neat.population[0];

    /**
     * When adding an external genome, population length increases by 1.
     */
    it('should increase population length by one', () => {
      // Arrange: external clone
      const before = neat.population.length;
      const external = parent.clone
        ? parent.clone()
        : Network.fromJSON(parent.toJSON());
      // Act: add genome through Neat API
      (neat as any).addGenome(external, [(parent as any)._id]);
      // Assert: population length incremented
      expect(neat.population.length).toBe(before + 1);
    });

    /**
     * When adding a genome with parents provided, the stored parents must match.
     */
    it('should attach provided parent ids to the added genome', () => {
      // Arrange: create an external genome and add it
      const external = parent.clone
        ? parent.clone()
        : Network.fromJSON(parent.toJSON());
      // Act: add genome
      (neat as any).addGenome(external, [(parent as any)._id]);
      // Assert: the most recently added genome has the expected parents
      const added = neat.population[neat.population.length - 1];
      expect((added as any)._parents).toEqual([(parent as any)._id]);
    });

    /**
     * When provided parents, addGenome should estimate depth as max(parent depths)+1.
     */
    it('should estimate depth based on parent depths', () => {
      // Arrange: create chain parents to increase depth
      const p1 = neat.population[0];
      // artificially create a deeper parent via spawnFromParent to produce different depths
      const p2 = (neat as any).spawnFromParent(p1, 1);
      // Register the spawned parent into neat so addGenome can resolve parent depths
      (neat as any).addGenome(p2, [(p1 as any)._id]);
      // Act: add an external genome with p1 and p2 as parents
      const external = p1.clone ? p1.clone() : Network.fromJSON(p1.toJSON());
      (neat as any).addGenome(external, [(p1 as any)._id, (p2 as any)._id]);
      // Assert: added genome depth is max(parent depths)+1
      const added = neat.population[neat.population.length - 1];
      const expectedDepth =
        Math.max((p1 as any)._depth ?? 0, (p2 as any)._depth ?? 0) + 1;
      expect((added as any)._depth).toBe(expectedDepth);
    });
  });
});
