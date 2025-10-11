import Network from '../../src/architecture/network';
import { maybePrune } from '../../src/architecture/network/network.prune';

interface PruningConfigSnapshot {
  start: number;
  end: number;
  frequency: number;
  targetSparsity: number;
  method: string;
  regrowFraction: number;
  lastPruneIter?: number;
}

const setInitialConnectionCount = (
  network: Network,
  connectionCount: number
): void => {
  Reflect.set(network, '_initialConnectionCount', connectionCount);
};

const setPruningConfig = (
  network: Network,
  config: PruningConfigSnapshot
): void => {
  Reflect.set(network, '_pruningConfig', config);
};

const getPruningConfigSnapshot = (
  network: Network
): PruningConfigSnapshot | null =>
  (Reflect.get(network, '_pruningConfig') as PruningConfigSnapshot | null) ??
  null;

/**
 * maybePrune early-exit branch (excessConnectionCount <= 0) should still stamp lastPruneIter
 * without modifying structure. Uses iteration=start so target sparsity = 0 (no removals).
 */
describe('Network.prune scheduled pruning', () => {
  describe('Scenario: early exit when already at/below target sparsity', () => {
    it('stamps lastPruneIter while keeping connection count unchanged', () => {
      // Arrange
      const net = new Network(2, 1, { seed: 21, enforceAcyclic: true });
      // Ensure at least one initial forward connection so _initialConnectionCount meaningful.
      if (net.connections.length === 0) {
        const input = net.nodes.find((n) => n.type === 'input')!;
        const output = net.nodes.find((n) => n.type === 'output')!;
        net.connect(input, output);
      }
      const initialConnCount = net.connections.length;
      setInitialConnectionCount(net, initialConnCount); // baseline captured by training normally
      setPruningConfig(net, {
        start: 0,
        end: 10,
        frequency: 1,
        targetSparsity: 0.5,
        method: 'magnitude',
        regrowFraction: 0,
      });
      // Act
      maybePrune.call(net, 0); // iteration == start -> targetSparsityNow ramps to 0 so no pruning
      // Assert
      const pruningConfig = getPruningConfigSnapshot(net);
      if (!pruningConfig) {
        throw new Error('Pruning config should be defined.');
      }
      expect(
        pruningConfig.lastPruneIter === 0 &&
          net.connections.length === initialConnCount
      ).toBe(true);
    });
  });
});
