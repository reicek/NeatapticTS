import Network, {
  exportVisualizationGraph,
  formatConstructSummary,
  toDot,
} from './network';

describe('architecture network root facade', () => {
  it('exports Network as a function', () => {
    expect(typeof Network).toBe('function');
  });

  it('exports formatConstructSummary as a function', () => {
    expect(typeof formatConstructSummary).toBe('function');
  });

  it('exports visualization serialization helpers as functions', () => {
    expect({
      exportVisualizationGraph: typeof exportVisualizationGraph,
      toDot: typeof toDot,
    }).toEqual({
      exportVisualizationGraph: 'function',
      toDot: 'function',
    });
  });
});
