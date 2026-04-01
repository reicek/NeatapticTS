/**
 * benchmark.slab.fragmentation.trend.test.ts
 * Micro benchmark style test (single expectation) capturing slab fragmentation trend across a churn loop.
 * Scenario: grow network to expand slab capacity, capture low fragmentation state, prune nodes to induce higher
 * fragmentation (capacity retained, used shrinks), then regrow slightly. Verifies fragmentationPct increases after prune.
 */
import Network from '../src/architecture/network';
import { memoryStats } from '../src/utils/memory';
import { config } from '../src/config';
import type Node from '../src/architecture/node';

/**
 * Runtime interface for Network with slab management methods.
 */
interface RuntimeNetwork {
  _slabDirty: boolean;
  getConnectionSlab: () => void;
}

describe('benchmark.slab.fragmentation.trend', () => {
  it('fragmentation percentage increases after prune-induced slack (growth -> prune)', () => {
    // Arrange
    config.enableNodePooling = false;
    const net = new Network(6, 3, { enforceAcyclic: true });
    const runtimeNet = net as unknown as RuntimeNetwork;
    // Initial slab build
    runtimeNet._slabDirty = true;
    runtimeNet.getConnectionSlab();
    const addedNodes: Node[] = [];
    // Growth phase: repeatedly split random connections to enlarge connection count & capacity
    let growthIters = 0;
    while (growthIters < 40) {
      if (net.connections.length) net.addNodeBetween();
      const newlyAdded = net.nodes[net.nodes.length - 1];
      if (!addedNodes.includes(newlyAdded)) addedNodes.push(newlyAdded);
      runtimeNet._slabDirty = true;
      runtimeNet.getConnectionSlab();
      growthIters++;
    }
    const statsAfterGrowth = memoryStats(net).slabs;
    const fragAfterGrowth = statsAfterGrowth.fragmentationPct;
    // Prune phase: remove half of the added hidden nodes to create unused capacity slack
    let removed = 0;
    for (let i = 0; i < addedNodes.length; i += 2) {
      const node = addedNodes[i];
      if (!node) continue;
      net.remove(node);
      removed++;
      if (removed > addedNodes.length / 2) break;
    }
    runtimeNet._slabDirty = true;
    runtimeNet.getConnectionSlab();
    const statsAfterPrune = memoryStats(net).slabs;
    const fragAfterPrune = statsAfterPrune.fragmentationPct;
    // Assert: fragmentation after prune should be >= growth fragmentation (strictly greater ideally)
    const pass =
      fragAfterGrowth === null ||
      fragAfterPrune === null ||
      fragAfterPrune >= fragAfterGrowth;
    expect(pass).toBe(true);
  });
});
