import Neat from '../../src/neat';
import Network from '../../src/architecture/network';
import { mutation } from '../../src/methods/mutation';

/** Tests for mutateAddConnReuse cycle prevention when _enforceAcyclic set. */
describe('Mutation add connection reuse (acyclic guard)', () => {
  describe('skips adding connection forming a cycle', () => {
    /** Fitness using connection count. */
    const fitness = (n: Network) => n.connections.length;
    const neat = new Neat(3, 1, fitness, {
      popsize: 2,
      seed: 880,
      mutation: [mutation.ADD_CONN],
    });
    let genome: any;
    beforeAll(async () => {
      await neat.evaluate();
      genome = neat.population[0];
      // Arrange: enforce acyclicity and create simple chain input->hidden->output
      (genome as any)._enforceAcyclic = true;
      // Force add a hidden node and connect chain
      const NodeClass = require('../../src/architecture/node').default;
      const hidden = new NodeClass('hidden');
      genome.nodes.splice(genome.nodes.length - genome.output, 0, hidden);
      genome.connect(
        genome.nodes.find((n: any) => n.type === 'input'),
        hidden,
        1
      );
      genome.connect(
        hidden,
        genome.nodes.find((n: any) => n.type === 'output'),
        1
      );
      // Attempt to create a back edge hidden->input via direct call (should be prevented in selection logic)
    });
    test('addConn reuse does not create cycle (no connection to earlier input)', () => {
      // Arrange: attempt many adds to increase chance of cycle candidate
      for (let i = 0; i < 5; i++) (neat as any)._mutateAddConnReuse(genome);
      // Act: search for illegal back edge to input
      const input = genome.nodes.find((n: any) => n.type === 'input');
      const illegal = genome.connections.some(
        (c: any) => c.from.type === 'hidden' && c.to === input
      );
      // Assert: no illegal back edge created
      expect(illegal).toBe(false);
    });
  });
});
