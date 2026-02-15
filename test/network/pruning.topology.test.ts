import Network from '../../src/architecture/network';
import Node from '../../src/architecture/node';
import {
  maybePrune,
  pruneToSparsity,
  getCurrentSparsity,
} from '../../src/architecture/network/network.prune.utils';
import {
  computeTopoOrder,
  hasPath,
} from '../../src/architecture/network/network.topology.utils';

type PruningMethod = 'magnitude' | 'snip';

type PruningConfig = {
  start: number;
  end: number;
  frequency: number;
  targetSparsity: number;
  method: PruningMethod;
  regrowFraction?: number;
  lastPruneIter?: number;
};

const setPruningConfig = (network: Network, config: PruningConfig) => {
  Reflect.set(network, '_pruningConfig', config);
};

const getPruningConfig = (network: Network): PruningConfig | undefined => {
  return Reflect.get(network, '_pruningConfig') as PruningConfig | undefined;
};

const captureInitialConnectionBaseline = (network: Network) => {
  Reflect.set(network, '_initialConnectionCount', network.connections.length);
};

const setTopoOrder = (network: Network, order: Node[] | null) => {
  Reflect.set(network, '_topoOrder', order);
};

const getTopoOrder = (network: Network): Node[] | null => {
  return Reflect.get(network, '_topoOrder') as Node[] | null;
};

const setTopoDirty = (network: Network, isDirty: boolean) => {
  Reflect.set(network, '_topoDirty', isDirty);
};

const getTopoDirty = (network: Network): boolean => {
  return Boolean(Reflect.get(network, '_topoDirty'));
};

const getNetworkRng = (network: Network): (() => number) => {
  return (Reflect.get(network, '_rand') as () => number) ?? Math.random;
};

/**
 * Pruning & topology scenario tests.
 * Each test uses a single expectation and AAA pattern.
 */

describe('Network.pruning & topology utilities', () => {
  describe('Scenario: maybePrune skips when outside window', () => {
    it('keeps connection count unchanged when iteration before start', () => {
      // Arrange
      const net = new Network(2, 1, { seed: 1, enforceAcyclic: true });
      setPruningConfig(net, {
        start: 5,
        end: 10,
        frequency: 1,
        targetSparsity: 0.5,
        method: 'magnitude',
      });
      captureInitialConnectionBaseline(net);
      const before = net.connections.length;
      // Act
      maybePrune.call(net, 0);
      // Assert
      expect(net.connections.length).toBe(before);
    });
  });

  describe('Scenario: maybePrune inside window but no pruning needed', () => {
    it('sets lastPruneIter without changing connection count when sparsity target yields no excess', () => {
      // Arrange
      const net = new Network(3, 2, { seed: 101, enforceAcyclic: true });
      setPruningConfig(net, {
        start: 0,
        end: 5,
        frequency: 1,
        targetSparsity: 0.8,
        method: 'magnitude',
      }); // iteration 0 progressFraction=0 => target NOW =0
      captureInitialConnectionBaseline(net);
      const before = net.connections.length;
      // Act
      maybePrune.call(net, 0); // inside window, but targetSparsityNow=0
      const configAfter = getPruningConfig(net);
      const unchanged =
        net.connections.length === before && configAfter?.lastPruneIter === 0;
      // Assert
      expect(unchanged).toBe(true);
    });
  });

  describe('Scenario: maybePrune prunes inside window', () => {
    it('reduces connection count inside active window', () => {
      // Arrange
      const net = new Network(3, 2, { seed: 2, enforceAcyclic: true });
      setPruningConfig(net, {
        start: 0,
        end: 2,
        frequency: 1,
        targetSparsity: 0.5,
        method: 'magnitude',
      });
      captureInitialConnectionBaseline(net);
      const before = net.connections.length;
      // Act
      maybePrune.call(net, 2); // end of window -> full target sparsity applied
      const after = net.connections.length;
      // Assert
      expect(after).toBeLessThan(before);
    });
  });

  describe('Scenario: maybePrune with SNIP method', () => {
    it('invokes pruning using snip ranking (connection count decreases)', () => {
      // Arrange
      const net = new Network(3, 2, { seed: 12, enforceAcyclic: true });
      setPruningConfig(net, {
        start: 0,
        end: 3,
        frequency: 1,
        targetSparsity: 0.4,
        method: 'snip',
      });
      captureInitialConnectionBaseline(net);
      const before = net.connections.length;
      // Act
      maybePrune.call(net, 3); // full progress for target sparsity
      const after = net.connections.length;
      // Assert
      expect(after).toBeLessThan(before);
    });
  });

  describe('Scenario: maybePrune with regrowth enabled', () => {
    it('results in connection count >= pruning-only scenario', () => {
      // Arrange
      const baseA = new Network(3, 2, { seed: 21, enforceAcyclic: true });
      const baseB = new Network(3, 2, { seed: 21, enforceAcyclic: true }); // identical topology & seed
      setPruningConfig(baseA, {
        start: 0,
        end: 0,
        frequency: 1,
        targetSparsity: 0.5,
        method: 'magnitude',
      });
      setPruningConfig(baseB, {
        start: 0,
        end: 0,
        frequency: 1,
        targetSparsity: 0.5,
        method: 'magnitude',
        regrowFraction: 1,
      });
      captureInitialConnectionBaseline(baseA);
      captureInitialConnectionBaseline(baseB);
      // Act
      maybePrune.call(baseA, 0); // prune only
      maybePrune.call(baseB, 0); // prune + regrow
      // Assert
      expect(baseB.connections.length >= baseA.connections.length).toBe(true);
    });
  });

  describe('Scenario: getCurrentSparsity baseline unset', () => {
    it('returns zero sparsity when baseline not captured', () => {
      // Arrange
      const net = new Network(2, 1, { seed: 3, enforceAcyclic: true });
      // Act
      const sparsity = getCurrentSparsity.call(net);
      // Assert
      expect(sparsity).toBe(0);
    });
  });

  describe('Scenario: getCurrentSparsity after pruning baseline set', () => {
    it('reports positive sparsity after manual disconnection', () => {
      // Arrange
      const net = new Network(2, 2, { seed: 31, enforceAcyclic: true });
      captureInitialConnectionBaseline(net);
      const toRemove = net.connections[0];
      net.disconnect(toRemove.from, toRemove.to);
      // Act
      const sparsity = getCurrentSparsity.call(net);
      // Assert
      expect(sparsity > 0).toBe(true);
    });
  });

  describe('Scenario: pruneToSparsity magnitude', () => {
    it('reduces connections toward target sparsity', () => {
      // Arrange
      const net = new Network(4, 2, { seed: 4, enforceAcyclic: true });
      const before = net.connections.length;
      // Act
      pruneToSparsity.call(net, 0.3, 'magnitude');
      const after = net.connections.length;
      // Assert
      expect(after).toBeLessThan(before);
    });
  });

  describe('Scenario: pruneToSparsity snip fallback (no gradient stats)', () => {
    it('still prunes using weight magnitude fallback', () => {
      // Arrange
      const net = new Network(4, 2, { seed: 5, enforceAcyclic: true });
      const before = net.connections.length;
      // Act
      pruneToSparsity.call(net, 0.2, 'snip');
      const after = net.connections.length;
      // Assert
      expect(after).toBeLessThan(before);
    });
  });

  describe('Scenario: pruneToSparsity trivial & clamp branches', () => {
    it('leaves connections unchanged when target sparsity <= 0', () => {
      // Arrange
      const net = new Network(3, 1, { seed: 41, enforceAcyclic: true });
      const before = net.connections.length;
      // Act
      pruneToSparsity.call(net, 0);
      const after = net.connections.length;
      // Assert
      expect(after).toBe(before);
    });
    it('does not remove all connections when target sparsity >= 1 (clamped)', () => {
      // Arrange
      const net = new Network(3, 1, { seed: 42, enforceAcyclic: true });
      // Act
      pruneToSparsity.call(net, 1);
      const remaining = net.connections.length;
      // Assert
      expect(remaining > 0).toBe(true);
    });
  });

  describe('Scenario: computeTopoOrder acyclic', () => {
    it('produces full topo order length when acyclic', () => {
      // Arrange
      const net = new Network(2, 2, { seed: 6, enforceAcyclic: true });
      // Act
      computeTopoOrder.call(net);
      const order = getTopoOrder(net);
      // Assert
      expect(order?.length).toBe(net.nodes.length);
    });
  });

  describe('Scenario: computeTopoOrder with cycle fallback', () => {
    it('falls back to raw node order length when cycle present', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 7, enforceAcyclic: true });
      // Add hidden node & create cycle hidden->output->hidden.
      const hidden = new Node('hidden', undefined, getNetworkRng(net));
      net.nodes.push(hidden);
      net.connect(net.nodes[0], hidden);
      net.connect(hidden, net.nodes[1]);
      net.connect(net.nodes[1], hidden); // back edge introduces cycle
      // Act
      computeTopoOrder.call(net);
      const order = getTopoOrder(net);
      // Assert
      expect(order?.length).toBe(net.nodes.length);
    });
  });

  describe('Scenario: computeTopoOrder non-acyclic mode', () => {
    it('clears cached topo order (sets to null) when acyclicity not enforced', () => {
      // Arrange
      const net = new Network(2, 1, { seed: 71, enforceAcyclic: false });
      setTopoOrder(net, [net.nodes[0]]);
      setTopoDirty(net, true);
      // Act
      computeTopoOrder.call(net);
      const cleared = getTopoOrder(net) === null && getTopoDirty(net) === false;
      // Assert
      expect(cleared).toBe(true);
    });
  });

  describe('Scenario: hasPath positive & negative', () => {
    it('returns true for reachable target', () => {
      // Arrange
      const net = new Network(2, 1, { seed: 8, enforceAcyclic: true });
      // Act
      const reachable = hasPath.call(
        net,
        net.nodes[0],
        net.nodes[net.nodes.length - 1],
      );
      // Assert
      expect(reachable).toBe(true);
    });
    it('returns false for unreachable target', () => {
      // Arrange
      const net = new Network(2, 1, { seed: 9, enforceAcyclic: true });
      // Disconnect all edges to isolate output.
      net.connections.slice().forEach((c) => net.disconnect(c.from, c.to));
      // Act
      const reachable = hasPath.call(
        net,
        net.nodes[0],
        net.nodes[net.nodes.length - 1],
      );
      // Assert
      expect(reachable).toBe(false);
    });
  });
});
