import {
  Architect,
  centerPositionedNodesInDrawableArea,
  config,
  Connection,
  exportVisualizationGraph,
  formatConstructSummary,
  Group,
  Layer,
  methods,
  multi,
  Neat,
  Network,
  Node,
  positionNetworkNodes,
  renderNetworkView,
  resolveNetworkVisualizationLayers,
  resolveNetworkVisualizationTopologyPlan,
  toDot,
} from './neataptic';

describe('neataptic root facade', () => {
  it('exports Neat as a function', () => {
    expect(typeof Neat).toBe('function');
  });

  it('exports Network as a function', () => {
    expect(typeof Network).toBe('function');
  });

  it('exports formatConstructSummary as a function', () => {
    expect(typeof formatConstructSummary).toBe('function');
  });

  it('exports Node as a function', () => {
    expect(typeof Node).toBe('function');
  });

  it('exports Layer as a function', () => {
    expect(typeof Layer).toBe('function');
  });

  it('exports Group as a function', () => {
    expect(typeof Group).toBe('function');
  });

  it('exports Connection as a function', () => {
    expect(typeof Connection).toBe('function');
  });

  it('exports Architect as a function', () => {
    expect(typeof Architect).toBe('function');
  });

  it('exports methods as an object', () => {
    expect(typeof methods).toBe('object');
  });

  it('exports config as an object', () => {
    expect(typeof config).toBe('object');
  });

  it('exports multi as an object', () => {
    expect(typeof multi).toBe('object');
  });

  it('exports visualization graph helpers as functions', () => {
    expect({
      exportVisualizationGraph: typeof exportVisualizationGraph,
      toDot: typeof toDot,
    }).toEqual({
      exportVisualizationGraph: 'function',
      toDot: 'function',
    });
  });

  it('exports shared network-view helpers as functions', () => {
    expect({
      centerPositionedNodesInDrawableArea:
        typeof centerPositionedNodesInDrawableArea,
      positionNetworkNodes: typeof positionNetworkNodes,
      renderNetworkView: typeof renderNetworkView,
      resolveNetworkVisualizationLayers:
        typeof resolveNetworkVisualizationLayers,
      resolveNetworkVisualizationTopologyPlan:
        typeof resolveNetworkVisualizationTopologyPlan,
    }).toEqual({
      centerPositionedNodesInDrawableArea: 'function',
      positionNetworkNodes: 'function',
      renderNetworkView: 'function',
      resolveNetworkVisualizationLayers: 'function',
      resolveNetworkVisualizationTopologyPlan: 'function',
    });
  });
});
