import Network from '../network';
import Node from '../../node';
import {
  asStandaloneProps,
  createGenerationContext,
} from './network.standalone.utils.setup';
import { buildNodeSumExpression } from './network.standalone.utils.graph';
import { SINGLE_TERM_FALLBACK } from './network.standalone.utils.types';

describe('network standalone graph utility chapter', () => {
  describe('buildNodeSumExpression', () => {
    describe('given an inbound connection source is missing its generated index', () => {
      it('skips the invalid term and falls back to the zero-expression literal', () => {
        // Arrange
        const network = new Network(1, 1, { seed: 7501 });
        const generationContext = createGenerationContext(
          asStandaloneProps(network),
        );
        const sourceNode = new Node('hidden') as Node & { index?: number };
        const targetNode = new Node('hidden');
        delete sourceNode.index;
        sourceNode.connect(targetNode);

        // Act
        const sumExpression = buildNodeSumExpression(
          generationContext,
          targetNode,
          0,
        );

        // Assert
        expect(sumExpression).toBe(SINGLE_TERM_FALLBACK);
      });
    });
  });
});
