import Node from '../../node';
import {
  createPathSearchContext,
  traversePathSearch,
} from './network.topology.path.utils';

describe('network topology path utilities chapter', () => {
  describe('traversePathSearch', () => {
    describe('given the search revisits one previously expanded node through a cycle', () => {
      it('skips the repeated node and reports that the target is unreachable', () => {
        // Arrange
        const hiddenNode = new Node('hidden');
        const sourceNode = new Node('input');
        const targetNode = new Node('output');

        sourceNode.connect(hiddenNode);
        hiddenNode.connect(sourceNode);
        const searchContext = createPathSearchContext(sourceNode, targetNode);

        // Act
        const targetIsReachable = traversePathSearch(searchContext);

        // Assert
        expect(targetIsReachable).toBe(false);
      });
    });

    describe('given the search sees only a self-loop on the current node', () => {
      it('ignores the self-loop and reports that the target is unreachable', () => {
        // Arrange
        const sourceNode = new Node('input');
        const targetNode = new Node('output');

        sourceNode.connections.out.push({ to: sourceNode } as never);
        const searchContext = createPathSearchContext(sourceNode, targetNode);

        // Act
        const targetIsReachable = traversePathSearch(searchContext);

        // Assert
        expect(targetIsReachable).toBe(false);
      });
    });
  });
});
