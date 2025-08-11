import Network from '../../src/architecture/network';
import * as methods from '../../src/methods/methods';

// Single expectation per test.

describe('Topological reorder cache', () => {
  const net = new Network(3, 2, { enforceAcyclic: true });
  test('computes topo order after structural mutation', () => {
    net.mutate(methods.mutation.ADD_NODE);
    net.mutate(methods.mutation.ADD_CONN);
    net.activate([0, 0, 0]);
    const order = (net as any)._topoOrder as any[] | null;
    expect(Array.isArray(order) && order.length === net.nodes.length).toBe(
      true
    );
  });
});
