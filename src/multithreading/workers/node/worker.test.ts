import Multi from '../../multi';

describe('node worker entrypoint chapter', () => {
  describe('process message handler', () => {
    async function loadMessageHandler(): Promise<(message: unknown) => void> {
      jest.resetModules();
      await import('./worker');

      const messageHandler = process.listeners('message').at(-1) as
        | ((message: unknown) => void)
        | undefined;

      if (!messageHandler) {
        throw new Error('Expected the worker module to register a process message handler.');
      }

      return messageHandler;
    }

    describe('given the worker has been initialized with a dataset and cost function', () => {
      it('evaluates the serialized network and sends a numeric score back', async () => {
        // Arrange
        const messageHandler = await loadMessageHandler();
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

    describe('given one evaluation message omits serialized state data', () => {
      it('logs the missing-data error and skips evaluation', async () => {
        // Arrange
        const messageHandler = await loadMessageHandler();
        const consoleErrorSpy = jest
          .spyOn(console, 'error')
          .mockImplementation(() => undefined);
        const testSerializedSetSpy = jest.spyOn(Multi, 'testSerializedSet');

        try {
          // Act
          messageHandler({ activations: [0], conns: [1, 2, 3] });

          // Assert
          expect({
            consoleErrorCalls: consoleErrorSpy.mock.calls,
            evaluationCalls: testSerializedSetSpy.mock.calls.length,
          }).toEqual({
            consoleErrorCalls: [['Missing required data for network evaluation']],
            evaluationCalls: 0,
          });
        } finally {
          consoleErrorSpy.mockRestore();
          testSerializedSetSpy.mockRestore();
          process.removeListener('message', messageHandler);
        }
      });
    });

    describe('given the worker evaluates a network without a parent-process send hook', () => {
      it('still computes the serialized score locally', async () => {
        // Arrange
        const messageHandler = await loadMessageHandler();
        const originalProcessSend = process.send;
        const serializedSet = Multi.serializeDataSet([
          { input: [2], output: [1] },
        ]);
        const serializedNetwork = [1, 1, 0, 0, 2, 0, -1, 0, 0.5, -1, -2];

        process.send = undefined;

        try {
          // Assert
          expect(() => {
            // Act
            messageHandler({ set: serializedSet, cost: 'mse' });
            messageHandler({
              activations: [0],
              states: [0],
              conns: serializedNetwork,
            });
          }).not.toThrow();
        } finally {
          process.send = originalProcessSend;
          process.removeListener('message', messageHandler);
        }
      });
    });
  });
});
