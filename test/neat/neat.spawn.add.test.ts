/**
 * Tests for Neat.spawnFromParent and Neat.addGenome behaviors.
 * These tests follow the AAA pattern and use single-expect assertions as required.
 */
import Neat from '../../src/neat';
import Network from '../../src/architecture/network';
import type {
  LineageTrackedNetwork,
  NeatLineageHarness,
} from '../../src/neat/neat.harness.types';

const withNeatLineage = (neat: Neat): NeatLineageHarness =>
  neat as NeatLineageHarness;
const withInternalGenome = (genome: Network): LineageTrackedNetwork =>
  genome as LineageTrackedNetwork;

/**
 * Top-level scenario grouping for spawn/add helpers of Neat.
 */
describe('Neat spawnFromParent and addGenome helpers', () => {
  /**
   * Factory to create a fresh Neat instance for each scenario.
   * We keep the fitness trivial since we only inspect bookkeeping side-effects.
   */
  const createNeatInstance = () =>
    new Neat(
      3,
      2,
      (network: Network) => {
        void network;
        return 0;
      },
      { popsize: 6 }
    );

  describe('spawnFromParent(parent, mutateCount)', () => {
    // Create a fresh neat and pick a parent from its initial pool
    const neat = createNeatInstance();
    const parent = withInternalGenome(neat.population[0]);
    const neatWithHelpers = withNeatLineage(neat);

    /**
     * Ensure spawnFromParent assigns a new unique genome id (distinct from parent).
     */
    it('should assign a new id distinct from parent', async () => {
      // Arrange: have neat and parent defined above
      // Act: spawn a child (await the async operation)
      const child = withInternalGenome(
        await neatWithHelpers.spawnFromParent(parent, 1)
      );
      // Assert: child id must not equal parent id
      expect(child._id).not.toBe(parent._id);
    });

    /**
     * Ensure spawnFromParent records the parent id in _parents array.
     */
    it('should set parent id in _parents', async () => {
      // Arrange & Act: spawn child (await the async operation)
      const child = withInternalGenome(
        await neatWithHelpers.spawnFromParent(parent, 1)
      );
      // Assert: the first parent id equals parent's id
      expect(child._parents).toEqual([parent._id]);
    });

    /**
     * Ensure the child's depth is parent's depth + 1 when lineage is enabled.
     */
    it('should set depth equal to parent.depth + 1', async () => {
      // Arrange
      const baseDepth = parent._depth ?? 0;
      // Act: spawn child (await the async operation)
      const child = withInternalGenome(
        await neatWithHelpers.spawnFromParent(parent, 1)
      );
      // Assert: child's depth increments parent's depth
      expect(child._depth).toBe(baseDepth + 1);
    });

    /**
     * Ensure structural invariants are preserved (child has at least one connection).
     */
    it('should ensure the spawned child has at least one connection', async () => {
      // Arrange/Act: spawn child (await the async operation)
      const child = await neatWithHelpers.spawnFromParent(parent, 1);
      // Assert: connections array length is greater than zero
      expect(child.connections.length).toBeGreaterThan(0);
    });
  });

  describe('addGenome(genome, parents?)', () => {
    // Create a fresh Neat for the addGenome scenarios
    const neat = createNeatInstance();
    const parent = withInternalGenome(neat.population[0]);
    const neatWithHelpers = withNeatLineage(neat);

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
      neatWithHelpers.addGenome(external, [parent._id]);
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
      neatWithHelpers.addGenome(external, [parent._id]);
      // Assert: the most recently added genome has the expected parents
      const added = withInternalGenome(
        neat.population[neat.population.length - 1]
      );
      expect(added._parents).toEqual([parent._id]);
    });

    /**
     * When provided parents, addGenome should estimate depth as max(parent depths)+1.
     */
    it('should estimate depth based on parent depths', async () => {
      // Arrange: create chain parents to increase depth
      const firstParent = withInternalGenome(neat.population[0]);
      // artificially create a deeper parent via spawnFromParent to produce different depths
      const secondParent = withInternalGenome(
        await neatWithHelpers.spawnFromParent(firstParent, 1)
      );
      // Register the spawned parent into neat so addGenome can resolve parent depths
      neatWithHelpers.addGenome(secondParent, [firstParent._id]);
      // Act: add an external genome with p1 and p2 as parents
      const external = firstParent.clone
        ? firstParent.clone()
        : Network.fromJSON(firstParent.toJSON());
      neatWithHelpers.addGenome(external, [firstParent._id, secondParent._id]);
      // Assert: added genome depth is max(parent depths)+1
      const added = withInternalGenome(
        neat.population[neat.population.length - 1]
      );
      const expectedDepth =
        Math.max(firstParent._depth ?? 0, secondParent._depth ?? 0) + 1;
      expect(added._depth).toBe(expectedDepth);
    });
  });
});
