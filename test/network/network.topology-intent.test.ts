import { Architect, Network } from '../../src/neataptic';

describe('Network topology intent', () => {
  it('marks perceptron builders as feed-forward intent', () => {
    const network = Architect.perceptron(3, 4, 1);

    expect(network.getTopologyIntent()).toBe('feed-forward');
  });

  it('enables acyclic enforcement for perceptron builders', () => {
    const network = Architect.perceptron(3, 4, 1);

    expect(Reflect.get(network, '_enforceAcyclic')).toBe(true);
  });

  it('accepts feed-forward intent through the public constructor', () => {
    const network = new Network(2, 1, { topologyIntent: 'feed-forward' });

    expect(network.getTopologyIntent()).toBe('feed-forward');
  });

  it('serializes topology intent in JSON payloads', () => {
    const network = Architect.perceptron(2, 3, 1);
    const jsonPayload = network.toJSON() as { topologyIntent?: string };

    expect(jsonPayload.topologyIntent).toBe('feed-forward');
  });

  it('restores topology intent during JSON deserialization', () => {
    const network = Architect.perceptron(2, 3, 1);
    const rebuiltNetwork = Network.fromJSON(
      network.toJSON() as Record<string, unknown>,
    );

    expect(rebuiltNetwork.getTopologyIntent()).toBe('feed-forward');
  });

  it('keeps the legacy acyclic setter aligned with topology intent', () => {
    const network = new Network(2, 1);
    network.setEnforceAcyclic(true);

    expect(network.getTopologyIntent()).toBe('feed-forward');
  });
});
