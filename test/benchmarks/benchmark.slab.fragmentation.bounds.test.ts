/**
 * benchmark.slab.fragmentation.bounds.test.ts
 * Ensures fragmentation percentage stays within [0,100] and responds to growth+prune cycles.
 */
import Network from '../../src/architecture/network';
import { memoryStats } from '../../src/utils/memory';
import { config } from '../../src/config';

describe('benchmark.slab.fragmentation.bounds', () => {
  it('fragmentationPct remains within bounds after growth/prune/regrow cycles', () => {
    config.enableNodePooling = false;
    const net = new Network(5, 2, { enforceAcyclic: true });
    const fragments: number[] = [];
    const record = () => {
      (net as any)._slabDirty = true;
      (net as any).getConnectionSlab();
      const frag = memoryStats(net).slabs.fragmentationPct;
      if (frag !== null) fragments.push(frag);
    };
    record();
    // Grow
    for (let i = 0; i < 25; i++) {
      if (net.connections.length) net.addNodeBetween();
      record();
    }
    // Prune some hidden nodes
    const hidden = net.nodes.filter((n: any) => n.type === 'hidden');
    for (let i = 0; i < hidden.length; i += 3) net.remove(hidden[i]);
    record();
    // Regrow a bit
    for (let i = 0; i < 10; i++) {
      if (net.connections.length) net.addNodeBetween();
      record();
    }
    const allInRange = fragments.every((f) => f >= 0 && f <= 100);
    expect(allInRange).toBe(true);
  });
});
