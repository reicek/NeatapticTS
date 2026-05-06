import {
  centerPositionedNodesInDrawableArea,
  positionNetworkNodes,
  renderNetworkView,
  resolveNetworkVisualizationLayers,
  resolveNetworkVisualizationTopologyPlan,
} from './visualization';

describe('visualization root facade', () => {
  it('exports the shared network-view helpers as functions', () => {
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
