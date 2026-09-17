import { resolveNetworkVisualizationLayers } from './visualization';

describe('visualization facade', () => {
  it('exports resolveNetworkVisualizationLayers from the topology utils', () => {
    expect(typeof resolveNetworkVisualizationLayers).toBe('function');
  });

  it('produces fallback layers from input/output sizes when no network is provided', () => {
    const layers = resolveNetworkVisualizationLayers(undefined, 2, 1);

    expect(layers.length).toBe(2);
    expect(layers[0]!.length).toBe(2);
    expect(layers[1]!.length).toBe(1);
  });

  it('places each input node in the input layer when no outputs are requested', () => {
    const layers = resolveNetworkVisualizationLayers(undefined, 2, 0);

    expect(layers.length).toBe(2);
    expect(layers[0]!.length).toBe(2);
    expect(layers[0]![0]!.index).toBe(0);
  });
});
