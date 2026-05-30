import { activationArrayPool } from '../../activationArrayPool/activationArrayPool';
import Network from '../network';
import { populatePooledOutputBufferFromNodes } from './network.activate.notrace.traversal.utils';

describe('network activate no-trace traversal utility chapter', () => {
  describe('populatePooledOutputBufferFromNodes', () => {
    it('writes the activated output value into the pooled buffer', () => {
      const network = new Network(1, 1, { seed: 5 });
      const outputNode = network.nodes.find(
        (nodeEntry) => nodeEntry.type === 'output',
      );
      const outputConnection = network.connections[0];

      if (!outputNode) {
        throw new Error('Expected one output node to exist');
      }

      if (!outputConnection) {
        throw new Error('Expected one output connection to exist');
      }

      outputNode.bias = 0;
      outputNode.squash = createIdentityActivation();
      outputConnection.weight = 1;

      const pooledOutputBuffer = activationArrayPool.acquire(1);

      try {
        populatePooledOutputBufferFromNodes({
          network,
          inputVector: [0.5],
          pooledOutputBuffer,
        });

        expect(pooledOutputBuffer[0]).toBeCloseTo(0.5);
      } finally {
        activationArrayPool.release(pooledOutputBuffer);
      }
    });
  });
});

function createIdentityActivation(): (
  value: number,
  derivate?: boolean,
) => number {
  return (value, derivate = false) => (derivate ? 1 : value);
}
