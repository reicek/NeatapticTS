/**
 * serialization.gater.test.ts
 * Validates that gater virtualization (symbol + bit flag) preserves gating relationships across
 * both compact tuple and verbose JSON serialization round-trips.
 * Single expectation style aggregating all checks.
 */
import Network from '../../src/architecture/network';
import {
  serialize as serializeTuple,
  deserialize as deserializeTuple,
  toJSONImpl,
  fromJSONImpl,
} from '../../src/architecture/network/network.serialize';

describe('serialization', () => {
  describe('gater virtualization round-trip', () => {
    it('should preserve gated connection count across serialization formats', () => {
      // Arrange: small network baseline.
      const net = new Network(2, 1);
      // Attempt to add a hidden node if API exists; otherwise reuse existing structure.
      const hidden = (net as any).addNode?.() || null;
      const nodes = (net as any).nodes;
      const input0 = nodes[0];
      const input1 = nodes[1];
      const output = nodes[nodes.length - 1];
      const hiddenNode =
        hidden ||
        nodes.find((n: any) => n !== input0 && n !== input1 && n !== output) ||
        output;
      (net as any).connect(input0, hiddenNode)[0]; // create a path
      const c2 = (net as any).connect(hiddenNode, output)[0]; // target to gate
      (net as any).gate(input1, c2); // apply gating relation

      const originalGated =
        (net as any).connections.filter((c: any) => !!c.gater).length +
        (net as any).selfconns.filter((c: any) => !!c.gater).length;

      // Act: tuple round-trip
      const tuple = serializeTuple.call(net);
      const netTuple = deserializeTuple(
        tuple,
        (net as any).input,
        (net as any).output
      );
      const tupleGated =
        (netTuple as any).connections.filter((c: any) => !!c.gater).length +
        (netTuple as any).selfconns.filter((c: any) => !!c.gater).length;

      // Act: verbose JSON round-trip
      const json = toJSONImpl.call(net) as any;
      const netJson = fromJSONImpl(json);
      const jsonGated =
        (netJson as any).connections.filter((c: any) => !!c.gater).length +
        (netJson as any).selfconns.filter((c: any) => !!c.gater).length;

      const pass =
        originalGated > 0 &&
        originalGated === tupleGated &&
        originalGated === jsonGated;
      expect(pass).toBe(true);
    });
  });
});
