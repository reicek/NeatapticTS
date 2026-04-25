import Network from '../network';
import { appendAllNodeComputationLines } from './network.standalone.utils.loop';
import {
  asStandaloneProps,
  createGenerationContext,
  seedNodeIndexesAndState,
} from './network.standalone.utils.setup';

describe('network standalone loop utility chapter', () => {
  describe('appendAllNodeComputationLines', () => {
    describe('given a node uses a non-identity mask value', () => {
      it('emits the multiplicative mask suffix in the activation line', () => {
        // Arrange
        const network = new Network(2, 1, { seed: 9902 });
        const outputNode = network.nodes.at(-1);
        if (!outputNode) {
          throw new Error('Expected an output node to exist');
        }

        outputNode.mask = 0.5;
        const generationContext = createGenerationContext(
          asStandaloneProps(network),
        );

        seedNodeIndexesAndState(generationContext);
        generationContext.activationNodeIndexes = [
          (outputNode as { index: number }).index,
        ];

        // Act
        appendAllNodeComputationLines(generationContext);
        const emittedActivationLine = generationContext.bodyLines.find(
          (line) => line.startsWith(`A[${outputNode.index}] =`),
        );

        // Assert
        expect(emittedActivationLine?.includes(' * 0.5;')).toBe(true);
      });
    });
  });
});