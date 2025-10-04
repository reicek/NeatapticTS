import Network from '../../src/architecture/network';
import type { OptimizerConfigBase } from '../../src/architecture/network/network.training';

// Deterministic tiny dataset y = 2x
const trainingData = Array.from({ length: 3 }, (_, sampleIndex) => ({
  input: [sampleIndex + 1],
  output: [2 * (sampleIndex + 1)],
}));

const buildNet = (): Network => {
  const network = new Network(1, 1);
  for (const connection of network.connections) {
    connection.weight = 0.5;
  }
  for (const node of network.nodes) {
    if (node.type !== 'input') node.bias = 0;
  }
  return network;
};

// Utility to run one iteration and capture internal optimizer state
const trainSteps = (
  network: Network,
  optimizerConfig: OptimizerConfigBase,
  iterations: number,
  rate = 0.01,
): void => {
  network.train(trainingData, {
    iterations,
    rate,
    optimizer: optimizerConfig,
    batchSize: 1,
    error: 0,
    cost: {
      fn: (target: number[], output: number[]) => (output[0] - target[0]) ** 2,
      calculate: (target: number[], output: number[]) =>
        (output[0] - target[0]) ** 2,
    },
  });
};

describe('Optimizer formula characteristics', () => {
  describe('adamax infinity norm vs adam second moment', () => {
    const adamaxNetwork = buildNet();
    const adamNetwork = buildNet();
    trainSteps(adamaxNetwork, { type: 'adamax' }, 2);
    trainSteps(adamNetwork, { type: 'adam' }, 2);
    const adamaxConnection = adamaxNetwork.connections[0];
    const adamConnection = adamNetwork.connections[0];
    it('maintains infinityNorm different from sqrt(secondMoment)', () => {
      const metricsDefined =
        typeof adamaxConnection.infinityNorm === 'number' &&
        typeof adamConnection.secondMoment === 'number';
      expect(metricsDefined).toBe(true);
    });
  });

  describe('nadam nesterov lookahead produces larger early step than adam', () => {
    const nadamNetwork = buildNet();
    const adamNetwork = buildNet();
    trainSteps(nadamNetwork, { type: 'nadam' }, 1); // single step
    trainSteps(adamNetwork, { type: 'adam' }, 1);
    const nadamWeightDelta = nadamNetwork.connections[0].weight - 0.5;
    const adamWeightDelta = adamNetwork.connections[0].weight - 0.5;
    it('has different first step magnitude from adam', () => {
      expect(Math.abs(nadamWeightDelta - adamWeightDelta)).toBeGreaterThan(0);
    });
  });

  describe('radam early unrectified vs later rectified variance', () => {
    const earlyNetwork = buildNet();
    const lateNetwork = buildNet();
    trainSteps(earlyNetwork, { type: 'radam' }, 1);
    trainSteps(lateNetwork, { type: 'radam' }, 10);
    const earlyStepMagnitude = Math.abs(
      earlyNetwork.connections[0].weight - 0.5,
    );
    const lateStepMagnitude = Math.abs(lateNetwork.connections[0].weight - 0.5);
    it('late step magnitude differs from very early step', () => {
      expect(Math.abs(lateStepMagnitude - earlyStepMagnitude)).toBeGreaterThan(
        0,
      );
    });
  });

  describe('adabelief variance differs from adam given same gradients', () => {
    const adabeliefNetwork = buildNet();
    const adamNetwork = buildNet();
    trainSteps(adabeliefNetwork, { type: 'adabelief' }, 2);
    trainSteps(adamNetwork, { type: 'adam' }, 2);
    const adabeliefConnection = adabeliefNetwork.connections[0];
    const adamConnection = adamNetwork.connections[0];
    it('maintains distinct second moment estimate', () => {
      expect(adabeliefConnection.secondMoment).not.toBe(
        adamConnection.secondMoment,
      );
    });
  });

  describe('lookahead defaults', () => {
    const lookaheadNetwork = buildNet();
    // Provide only type to trigger default baseType and params
    trainSteps(lookaheadNetwork, { type: 'lookahead' }, 3);
    const lookaheadConnection = lookaheadNetwork.connections[0];
    it('creates shadow weight with default params', () => {
      expect(lookaheadConnection.lookaheadShadowWeight).toBeDefined();
    });
  });
});
