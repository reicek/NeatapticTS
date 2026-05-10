import Network, {
  createNeatParallelPopulationEvaluator,
  createInferencePredictor,
  detectInferenceWorkerCapabilities,
  evaluateInWorkers,
  extractNetworkInferenceIR,
  getTransferList,
  openInferenceChannel,
  openSharedInferenceWorker,
  ParallelInferencePool,
  exportPortableInferencePayload,
  exportTransferableInferencePayload,
  exportVisualizationGraph,
  formatConstructSummary,
  INFERENCE_ACTIVATION_TABLE,
  resolveAutoInferenceTransport,
  resolveBrowserWorkerAssetUrl,
  SHARED_INFERENCE_REQUIRES_CROSS_ORIGIN_ISOLATION,
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

  it('exports worker-payload helpers for inference transport', () => {
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
});
