import Neat from '../../src/neat';
import Network from '../../src/architecture/network';
import Node from '../../src/architecture/node';
import { mutation } from '../../src/methods/mutation';

/** Tests for mutateAddConnReuse cycle prevention when _enforceAcyclic set. */
describe('Mutation add connection reuse (acyclic guard)', () => {
  describe('skips adding connection forming a cycle', () => {
    /** Fitness using connection count. */
    const fitness = (network: Network) => network.connections.length;
    const neat = new Neat(3, 1, fitness, {
      popsize: 2,
      seed: 880,
      mutation: [mutation.ADD_CONN],
    });
    const neatWithInternals = neat as unknown as {
      _mutateAddConnReuse: (network: Network) => void;
    };
    let genome: Network;

    beforeAll(async () => {
      await neat.evaluate();
      genome = neat.population[0];
      // Step 1: enforce acyclicity and create simple chain input->hidden->output
      (genome as unknown as { _enforceAcyclic?: boolean })._enforceAcyclic = true;
      // Step 2: insert hidden node prior to output nodes
      const hiddenNode = new Node('hidden');
      genome.nodes.splice(genome.nodes.length - genome.output, 0, hiddenNode);
      // Step 3: connect input -> hidden and hidden -> output
  const inputNode = genome.nodes.find((node: Node) => node.type === 'input');
  const outputNode = genome.nodes.find((node: Node) => node.type === 'output');
      if (inputNode !== undefined && outputNode !== undefined) {
        genome.connect(inputNode, hiddenNode, 1);
        genome.connect(hiddenNode, outputNode, 1);
      }
      // Attempt to create a back edge hidden->input via direct call (should be prevented in selection logic)
    });

    test('addConn reuse does not create cycle (no connection to earlier input)', () => {
      // Arrange: attempt many adds to increase chance of cycle candidate
      for (let attemptIndex = 0; attemptIndex < 5; attemptIndex += 1) {
        neatWithInternals._mutateAddConnReuse(genome);
      }
      // Act: search for illegal back edge to input
      const inputNode = genome.nodes.find((node: Node) => node.type === 'input');
      const illegalBackEdgeExists =
        inputNode !== undefined &&
        genome.connections.some(
          (connection: (typeof genome.connections)[number]) =>
            connection.from.type === 'hidden' && connection.to === inputNode,
        );
      // Assert: no illegal back edge created
      expect(illegalBackEdgeExists).toBe(false);
    });
  });
});
