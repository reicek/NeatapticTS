import {
  Architect,
  centerPositionedNodesInDrawableArea,
  config,
  Connection,
  createNeatParallelPopulationEvaluator,
  createInferencePredictor,
  detectInferenceWorkerCapabilities,
  evaluateCandidate,
  evaluateInWorkers,
  extractNetworkInferenceIR,
  fineTuneVector,
  fromParameterVector,
  getTransferList,
  openInferenceChannel,
  openSharedInferenceWorker,
  ParallelInferencePool,
  exportPortableInferencePayload,
  exportTransferableInferencePayload,
  exportVisualizationGraph,
  formatConstructSummary,
  Group,
  INFERENCE_ACTIVATION_TABLE,
  Layer,
  methods,
  multi,
  Neat,
  Network,
  Node,
  positionNetworkNodes,
  renderNetworkView,
  resolveAutoInferenceTransport,
  resolveBrowserWorkerAssetUrl,
  resolveNetworkVisualizationLayers,
  resolveNetworkVisualizationTopologyPlan,
  SHARED_INFERENCE_REQUIRES_CROSS_ORIGIN_ISOLATION,
  toDot,
  toParameterVector,
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

  it('exports worker-payload helpers as part of the public facade', () => {
    expect({
      createNeatParallelPopulationEvaluator:
        typeof createNeatParallelPopulationEvaluator,
      createInferencePredictor: typeof createInferencePredictor,
      detectInferenceWorkerCapabilities:
        typeof detectInferenceWorkerCapabilities,
      evaluateInWorkers: typeof evaluateInWorkers,
      extractNetworkInferenceIR: typeof extractNetworkInferenceIR,
      getTransferList: typeof getTransferList,
      openInferenceChannel: typeof openInferenceChannel,
      openSharedInferenceWorker: typeof openSharedInferenceWorker,
      ParallelInferencePool: typeof ParallelInferencePool,
      resolveAutoInferenceTransport: typeof resolveAutoInferenceTransport,
      resolveBrowserWorkerAssetUrl: typeof resolveBrowserWorkerAssetUrl,
      exportPortableInferencePayload: typeof exportPortableInferencePayload,
      exportTransferableInferencePayload:
        typeof exportTransferableInferencePayload,
      inferenceActivationTableIsArray: Array.isArray(
        INFERENCE_ACTIVATION_TABLE,
      ),
      sharedInferenceRequiresCrossOriginIsolation:
        SHARED_INFERENCE_REQUIRES_CROSS_ORIGIN_ISOLATION,
    }).toEqual({
      createNeatParallelPopulationEvaluator: 'function',
      createInferencePredictor: 'function',
      detectInferenceWorkerCapabilities: 'function',
      evaluateInWorkers: 'function',
      extractNetworkInferenceIR: 'function',
      getTransferList: 'function',
      openInferenceChannel: 'function',
      openSharedInferenceWorker: 'function',
      ParallelInferencePool: 'function',
      resolveAutoInferenceTransport: 'function',
      resolveBrowserWorkerAssetUrl: 'function',
      exportPortableInferencePayload: 'function',
      exportTransferableInferencePayload: 'function',
      inferenceActivationTableIsArray: true,
      sharedInferenceRequiresCrossOriginIsolation: true,
    });
  });

  it('exports vector and hybrid helpers as part of the public facade', () => {
    expect({
      evaluateCandidate: typeof evaluateCandidate,
      fineTuneVector: typeof fineTuneVector,
      fromParameterVector: typeof fromParameterVector,
      toParameterVector: typeof toParameterVector,
    }).toEqual({
      evaluateCandidate: 'function',
      fineTuneVector: 'function',
      fromParameterVector: 'function',
      toParameterVector: 'function',
    });
  });
});
