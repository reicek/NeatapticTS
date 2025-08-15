import * as workerModule from '../../src/multithreading/workers/node/worker';
import Multi from '../../src/multithreading/multi';

// Helper to patch process.send for tests
const setProcessSend = (fn: any) => {
  (process as any).send = fn;
};

describe('node worker process handler', () => {
  it('handles dataset init and evaluation messages', () => {
    // Ensure module is loaded to attach the listener
    expect(workerModule).toBeDefined();

    const set = Multi.serializeDataSet([{ input: [2], output: [1] }]);

    // Grab the current 'message' listeners
    const listeners = process.listeners('message');
    expect(listeners.length).toBeGreaterThan(0);
    const handler = listeners[listeners.length - 1] as Function;

    // Initialize cost and dataset
    handler({ set, cost: 'mse' });

    // Prepare a minimal serialized network (1 input -> 1 output, identity)
    const data: number[] = [1, 1];
    data.push(0, 0, 2, 0, -1); // node 0
    data.push(0, 0.5, -1); // one incoming
    data.push(-2);

    const sendSpy = jest.fn();
    setProcessSend(sendSpy);

    handler({ activations: [0], states: [0], conns: data });

    expect(sendSpy).toHaveBeenCalled();
    const arg = sendSpy.mock.calls[0][0];
    expect(typeof arg).toBe('number');
  });
});
