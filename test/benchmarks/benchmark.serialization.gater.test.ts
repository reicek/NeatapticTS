/**
 * serialization.gater.test.ts
 * Validates that gater virtualization (symbol + bit flag) preserves gating relationships across
 * both compact tuple and verbose JSON serialization round-trips.
 * Single expectation style aggregating all checks.
 */
import Network from '../../src/architecture/network';
import type Node from '../../src/architecture/node';
import type Connection from '../../src/architecture/connection';
import {
  serialize as serializeTuple,
  deserialize as deserializeTuple,
  toJSONImpl,
  fromJSONImpl,
} from '../../src/architecture/network/network.serialize';

/**
 * Runtime interface for Network with internal properties and methods.
 */
interface RuntimeNetwork {
  addNode?: () => Node;
  nodes: Node[];
  connections: Connection[];
  selfconns: Connection[];
  connect: (from: Node, to: Node) => Connection[];
  gate: (node: Node, connection: Connection) => void;
  input: number;
  output: number;
}

/**
 * Runtime interface for Connection with gater property.
 */
interface RuntimeConnection {
  gater: Node | null;
}

describe('serialization', () => {
  describe('gater virtualization round-trip', () => {
    it('should preserve gated connection count across serialization formats', () => {
      // Arrange: small network baseline.
      const net = new Network(2, 1);
      const runtimeNet = (net as unknown) as RuntimeNetwork;
      // Attempt to add a hidden node if API exists; otherwise reuse existing structure.
      const hidden = runtimeNet.addNode?.() || null;
      const nodes = runtimeNet.nodes;
      const input0 = nodes[0];
      const input1 = nodes[1];
      const output = nodes.at(-1)!;
      const hiddenNode =
        hidden ||
        nodes.find((n) => n !== input0 && n !== input1 && n !== output) ||
        output;
      const c1 = runtimeNet.connect(input0, hiddenNode)[0]; // create a path
      const c2 = runtimeNet.connect(hiddenNode, output)[0]; // target to gate
      runtimeNet.gate(input1, c2); // apply gating relation
      // Ensure c1 is used
      expect(c1).toBeDefined();

      const originalGated =
        runtimeNet.connections.filter(
          (c) => !!((c as unknown) as RuntimeConnection).gater
        ).length +
        runtimeNet.selfconns.filter(
          (c) => !!((c as unknown) as RuntimeConnection).gater
        ).length;

      // Act: tuple round-trip
      const tuple = serializeTuple.call(net);
      const netTuple = deserializeTuple(
        tuple,
        runtimeNet.input,
        runtimeNet.output
      );
      const runtimeTuple = (netTuple as unknown) as RuntimeNetwork;
      const tupleGated =
        runtimeTuple.connections.filter(
          (c) => !!((c as unknown) as RuntimeConnection).gater
        ).length +
        runtimeTuple.selfconns.filter(
          (c) => !!((c as unknown) as RuntimeConnection).gater
        ).length;

      // Act: verbose JSON round-trip
      const json = toJSONImpl.call(net);
      const netJson = fromJSONImpl(json);
      const runtimeJson = (netJson as unknown) as RuntimeNetwork;
      const jsonGated =
        runtimeJson.connections.filter(
          (c) => !!((c as unknown) as RuntimeConnection).gater
        ).length +
        runtimeJson.selfconns.filter(
          (c) => !!((c as unknown) as RuntimeConnection).gater
        ).length;

      const pass =
        originalGated > 0 &&
        originalGated === tupleGated &&
        originalGated === jsonGated;
      expect(pass).toBe(true);
    });
  });
});
