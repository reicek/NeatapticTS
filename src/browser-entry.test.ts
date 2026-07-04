import {
  activateGPU,
  Architect,
  batchActivate,
  buildOverheadArtifact,
  computeOverheadBreakdown,
  config,
  Connection,
  formatConstructSummary,
  Group,
  GpuProfilingTimer,
  Layer,
  methods,
  Neat,
  Network,
  Node,
  prepareActivationContext,
  profileGPUActivation,
  rankWeakPoints,
} from './browser-entry';
import * as browserEntry from './browser-entry';
import { createMockGPUDevice } from './architecture/network/gpu/__mocks__/gpu.mock';

describe('browser entry facade', () => {
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

  it('does not export multi', () => {
    expect('multi' in browserEntry).toBe(false);
  });
});

function buildGPUEligibleMLP(
  inputCount: number,
  hiddenCounts: number[],
  outputCount: number,
) {
  const network = Network.createMLP(inputCount, hiddenCounts, outputCount);
  for (let index = 0; index < network.nodes.length; index++) {
    network.nodes[index].index = index;
  }
  return network;
}

describe('browser GPU and profiling exports', () => {
  it('exports activateGPU as a function', () => {
    expect(typeof activateGPU).toBe('function');
  });

  it('exports batchActivate as a function', () => {
    expect(typeof batchActivate).toBe('function');
  });

  it('exports profileGPUActivation as a function', () => {
    expect(typeof profileGPUActivation).toBe('function');
  });

  it('exports prepareActivationContext as a function', () => {
    expect(typeof prepareActivationContext).toBe('function');
  });

  it('exports computeOverheadBreakdown as a function', () => {
    expect(typeof computeOverheadBreakdown).toBe('function');
  });

  it('exports rankWeakPoints as a function', () => {
    expect(typeof rankWeakPoints).toBe('function');
  });

  it('exports buildOverheadArtifact as a function', () => {
    expect(typeof buildOverheadArtifact).toBe('function');
  });

  it('exports GpuProfilingTimer as a class', () => {
    expect(typeof GpuProfilingTimer).toBe('function');
  });

  it('activates a network with a mock GPU device', async () => {
    const network = buildGPUEligibleMLP(2, [3], 1);
    const output = await activateGPU(
      createMockGPUDevice(),
      network,
      [0.1, 0.2],
    );

    expect(output).toHaveLength(1);
  });

  it('runs batch activation on a mock GPU device', async () => {
    const network = buildGPUEligibleMLP(2, [3], 1);
    const { outputs } = await batchActivate(
      createMockGPUDevice(),
      [network, network],
      new Float32Array([0.1, 0.2, 0.3, 0.4]),
    );

    expect(outputs).toBeInstanceOf(Float32Array);
  });

  it('returns one row per network from batch activation', async () => {
    const network = buildGPUEligibleMLP(2, [3], 1);
    const { rowCount } = await batchActivate(
      createMockGPUDevice(),
      [network, network],
      new Float32Array([0.1, 0.2, 0.3, 0.4]),
    );

    expect(rowCount).toBe(2);
  });

  it('returns one column per output node from batch activation', async () => {
    const network = buildGPUEligibleMLP(2, [3], 1);
    const { colCount } = await batchActivate(
      createMockGPUDevice(),
      [network, network],
      new Float32Array([0.1, 0.2, 0.3, 0.4]),
    );

    expect(colCount).toBe(1);
  });

  it('profiles a GPU activation on a mock device', async () => {
    const network = buildGPUEligibleMLP(2, [3], 1);
    const result = await profileGPUActivation(
      createMockGPUDevice(),
      network,
      [0.1, 0.2],
    );

    expect(result.success).toBe(true);
  });

  it('prepares an activation context through the browser export', () => {
    const network = buildGPUEligibleMLP(2, [3], 1);
    const context = prepareActivationContext(network);
    context.restore();

    expect(context.index).toBe(0);
  });

  it('computes an overhead breakdown through the browser export', () => {
    const breakdown = computeOverheadBreakdown({
      pipeline: 10,
      bufferUpload: 5,
    });

    expect(breakdown[0].name).toBe('pipeline');
  });

  it('ranks weak points through the browser export', () => {
    const ranked = rankWeakPoints([
      { name: 'pipeline', ms: 10, pct: 50 },
      { name: 'bufferUpload', ms: 5, pct: 25 },
    ]);

    expect(ranked[0].name).toBe('pipeline');
  });

  it('builds an overhead artifact through the browser export', () => {
    const artifact = buildOverheadArtifact([], {});

    expect((artifact.summary as Record<string, unknown>).tierCount).toBe(0);
  });

  it('accumulates durations with the exported profiling timer', () => {
    const timer = new GpuProfilingTimer();
    timer.start('pipeline');
    timer.stop('pipeline');

    expect(timer.get('pipeline')).toBeGreaterThanOrEqual(0);
  });
});
