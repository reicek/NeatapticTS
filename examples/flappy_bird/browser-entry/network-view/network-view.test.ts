import Network from '../../../../src/architecture/network';
import Node from '../../../../src/architecture/node';
import { resolveNetworkArchitectureLabel } from './network-view';

describe('resolveNetworkArchitectureLabel', () => {
  it('prefers explicit runtime IO role sizes over caller-provided fallback hints', () => {
    const network = new Network(3, 2, { seed: 812 });
    const architectureLabel = resolveNetworkArchitectureLabel(network, 38, 99);

    expect(architectureLabel).toMatch(/^3 \| - \| 2\n\(\d+ nodes, \d+ connections\)$/);
  });

  it('adds a recurrent scheduling line when the runtime advertises recurrent execution', () => {
    const network = new Network(1, 1, {
      seed: 813,
      enforceAcyclic: false,
    });
    const inputNode = network.nodes[0];
    const outputNode = network.nodes[1];
    const hiddenNode = new Node('hidden');

    network.nodes = [inputNode, hiddenNode, outputNode];
    network.connections.slice().forEach((connection) => {
      network.disconnect(connection.from, connection.to);
    });
    network.connect(inputNode, hiddenNode);
    network.connect(hiddenNode, hiddenNode);
    network.connect(hiddenNode, outputNode);
    network.activate([1]);

    const architectureLabel = resolveNetworkArchitectureLabel(network, 1, 1);

    expect(architectureLabel).toContain('schedule: recurrent via compiled schedule');
  });

  it('adds a warning line when acyclic scheduling falls back because of a cycle', () => {
    const network = new Network(1, 1, {
      seed: 814,
      enforceAcyclic: true,
    });
    const inputNode = network.nodes[0];
    const outputNode = network.nodes[1];
    const hiddenNode = new Node('hidden');

    network.nodes.push(hiddenNode);
    inputNode.connect(hiddenNode);
    hiddenNode.connect(outputNode);
    outputNode.connect(hiddenNode);
    Network.rebuildConnections(network);
    network.activate([1]);

    const architectureLabel = resolveNetworkArchitectureLabel(network, 1, 1);

    expect(architectureLabel).toContain('warning: acyclic via cycle fallback');
  });
});