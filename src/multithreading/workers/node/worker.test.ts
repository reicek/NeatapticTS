import Multi from '../../multi';

describe('node worker entrypoint chapter', () => {
  describe('process message handler', () => {
    describe('given the worker has been initialized with a dataset and cost function', () => {
      it('evaluates the serialized network and sends a numeric score back', async () => {
        // Arrange
        await import('./worker');
        const messageHandler = process.listeners('message').at(-1) as
          | ((message: unknown) => void)
          | undefined;
        const originalProcessSend = process.send;
        const sendSpy = jest.fn();
        const serializedSet = Multi.serializeDataSet([
          { input: [2], output: [1] },
        ]);
        const serializedNetwork = [1, 1, 0, 0, 2, 0, -1, 0, 0.5, -1, -2];

        process.send = sendSpy as typeof process.send;

        try {
          // Act
          messageHandler?.({ set: serializedSet, cost: 'mse' });
          messageHandler?.({
            activations: [0],
            states: [0],
            conns: serializedNetwork,
          });
        } finally {
          process.send = originalProcessSend;
          if (messageHandler) {
            process.removeListener('message', messageHandler);
          }
        }

        // Assert
        expect(sendSpy.mock.calls[0]).toEqual([expect.any(Number)]);
      });
    });
  });
});
