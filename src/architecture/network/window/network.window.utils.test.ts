import { forwardWindowed, forwardWindowedAsync } from './network.window.utils';

type WindowTestNetwork = {
  activate: jest.Mock<number[], [number[], boolean?]>;
  input: number;
};

function createWindowTestNetwork(): WindowTestNetwork {
  return {
    activate: jest.fn((inputVector: number[]) => [inputVector[0] ?? 0]),
    input: 1,
  };
}

describe('network window utility chapter', () => {
  describe('forwardWindowed()', () => {
    describe('given the top-level input is not an array', () => {
      it('throws the collection-shape error', () => {
        // Arrange
        const network = createWindowTestNetwork();
        const activateInvalidSequence = () =>
          forwardWindowed.call(network as never, 'nope' as never);

        // Assert
        expect(activateInvalidSequence).toThrow(
          'inputs must be an array of input arrays',
        );
      });
    });

    describe('given one input row has the wrong width', () => {
      it('reports the received row width in the mismatch error', () => {
        // Arrange
        const network = createWindowTestNetwork();
        const activateInvalidSequence = () =>
          forwardWindowed.call(network as never, [[1, 2]]);

        // Assert
        expect(activateInvalidSequence).toThrow(
          'Input[0] size mismatch: expected 1, got 2',
        );
      });
    });

    describe('given one input row is undefined', () => {
      it('reports the received row width as undefined', () => {
        // Arrange
        const network = createWindowTestNetwork();
        const activateInvalidSequence = () =>
          forwardWindowed.call(network as never, [undefined] as never);

        // Assert
        expect(activateInvalidSequence).toThrow(
          'Input[0] size mismatch: expected 1, got undefined',
        );
      });
    });

    describe('given one input row is a Float32Array with the correct width', () => {
      it('accepts the typed-array row without treating it as an input mismatch', () => {
        // Arrange
        const network = createWindowTestNetwork();
        const activateTypedSequence = () =>
          forwardWindowed.call(
            network as never,
            [new Float32Array([1])] as unknown as number[][],
          );

        // Assert
        expect(activateTypedSequence).not.toThrow();
      });
    });

    describe('given output collection is disabled', () => {
      it('streams window callbacks without retaining the full output matrix', () => {
        // Arrange
        const network = createWindowTestNetwork();
        const windowSizes: number[] = [];

        // Act
        const outputRows = forwardWindowed.call(
          network as never,
          [[1], [2], [3]],
          {
            collectOutputs: false,
            onWindow(windowChunk) {
              windowSizes.push(windowChunk.outputs.length);
            },
            windowSize: 2,
          },
        );

        // Assert
        expect({ outputRows, windowSizes }).toEqual({
          outputRows: [],
          windowSizes: [2, 1],
        });
      });
    });

    describe('given the browser default window size is used', () => {
      it('emits browser-sized chunks when no explicit window size is provided', () => {
        // Arrange
        const network = createWindowTestNetwork();
        const originalWindow = Reflect.get(globalThis, 'window');
        const windowSizes: number[] = [];
        const inputSequence = Array.from({ length: 33 }, () => [1]);

        Reflect.set(globalThis, 'window', {});

        try {
          // Act
          forwardWindowed.call(network as never, inputSequence, {
            collectOutputs: false,
            onWindow(windowChunk) {
              windowSizes.push(windowChunk.outputs.length);
            },
          });

          // Assert
          expect(windowSizes).toEqual([32, 1]);
        } finally {
          Reflect.set(globalThis, 'window', originalWindow);
        }
      });
    });

    describe('given the Node default window size is used', () => {
      it('emits larger Node-sized chunks when no explicit window size is provided', () => {
        // Arrange
        const network = createWindowTestNetwork();
        const originalWindow = Reflect.get(globalThis, 'window');
        const windowSizes: number[] = [];
        const inputSequence = Array.from({ length: 129 }, () => [1]);

        Reflect.set(globalThis, 'window', undefined);

        try {
          // Act
          forwardWindowed.call(network as never, inputSequence, {
            collectOutputs: false,
            onWindow(windowChunk) {
              windowSizes.push(windowChunk.outputs.length);
            },
          });

          // Assert
          expect(windowSizes).toEqual([128, 1]);
        } finally {
          Reflect.set(globalThis, 'window', originalWindow);
        }
      });
    });
  });

  describe('forwardWindowedAsync()', () => {
    describe('given no async options are provided', () => {
      it('uses the default async window configuration without changing output order', async () => {
        // Arrange
        const network = createWindowTestNetwork();

        // Act
        const outputRows = await forwardWindowedAsync.call(network as never, [
          [1],
          [2],
          [3],
        ]);

        // Assert
        expect(outputRows).toEqual([[1], [2], [3]]);
      });
    });

    describe('given async window callbacks stream without retaining outputs', () => {
      it('awaits each emitted chunk while returning an empty collected matrix', async () => {
        // Arrange
        const network = createWindowTestNetwork();
        const windowRanges: number[][] = [];

        // Act
        const outputRows = await forwardWindowedAsync.call(
          network as never,
          [[1], [2], [3]],
          {
            collectOutputs: false,
            onWindow: async (windowChunk) => {
              windowRanges.push([
                windowChunk.startIndex,
                windowChunk.endIndexExclusive,
              ]);
            },
            windowSize: 2,
          },
        );

        // Assert
        expect({ outputRows, windowRanges }).toEqual({
          outputRows: [],
          windowRanges: [
            [0, 2],
            [2, 3],
          ],
        });
      });
    });

    describe('given an explicit async yield hook is provided', () => {
      it('yields after the configured number of completed windows while preserving output order', async () => {
        // Arrange
        const network = createWindowTestNetwork();
        let yieldCount = 0;

        // Act
        const outputRows = await forwardWindowedAsync.call(
          network as never,
          [[1], [2], [3], [4]],
          {
            windowSize: 2,
            yieldAfterWindows: 1,
            yieldControl: async () => {
              yieldCount += 1;
            },
          },
        );

        // Assert
        expect({ outputRows, yieldCount }).toEqual({
          outputRows: [[1], [2], [3], [4]],
          yieldCount: 1,
        });
      });
    });

    describe('given the browser default scheduler uses requestAnimationFrame', () => {
      it('yields through animation frames when no explicit yield hook is supplied', async () => {
        // Arrange
        const network = createWindowTestNetwork();
        const originalWindow = Reflect.get(globalThis, 'window');
        const originalRequestAnimationFrame = Reflect.get(
          globalThis,
          'requestAnimationFrame',
        );
        let frameCount = 0;

        Reflect.set(globalThis, 'window', {});
        Reflect.set(globalThis, 'requestAnimationFrame', ((
          callback: FrameRequestCallback,
        ) => {
          frameCount += 1;
          callback(0);
          return frameCount;
        }) as unknown as typeof requestAnimationFrame);

        try {
          // Act
          const outputRows = await forwardWindowedAsync.call(
            network as never,
            [[1], [2], [3]],
            {
              windowSize: 2,
            },
          );

          // Assert
          expect({ frameCount, outputRows }).toEqual({
            frameCount: 1,
            outputRows: [[1], [2], [3]],
          });
        } finally {
          Reflect.set(globalThis, 'window', originalWindow);
          Reflect.set(
            globalThis,
            'requestAnimationFrame',
            originalRequestAnimationFrame,
          );
        }
      });
    });

    describe('given the browser falls back to timer turns', () => {
      it('yields through setTimeout when animation frames are unavailable', async () => {
        // Arrange
        const network = createWindowTestNetwork();
        const originalWindow = Reflect.get(globalThis, 'window');
        const originalRequestAnimationFrame = Reflect.get(
          globalThis,
          'requestAnimationFrame',
        );
        const originalSetTimeout = Reflect.get(globalThis, 'setTimeout');
        let timerCount = 0;

        Reflect.set(globalThis, 'window', {});
        Reflect.set(globalThis, 'requestAnimationFrame', undefined);
        Reflect.set(globalThis, 'setTimeout', ((callback: TimerHandler) => {
          timerCount += 1;

          if (typeof callback === 'function') {
            callback();
          }

          return timerCount;
        }) as unknown as typeof setTimeout);

        try {
          // Act
          const outputRows = await forwardWindowedAsync.call(
            network as never,
            [[1], [2], [3]],
            {
              windowSize: 2,
            },
          );

          // Assert
          expect({ outputRows, timerCount }).toEqual({
            outputRows: [[1], [2], [3]],
            timerCount: 1,
          });
        } finally {
          Reflect.set(globalThis, 'window', originalWindow);
          Reflect.set(
            globalThis,
            'requestAnimationFrame',
            originalRequestAnimationFrame,
          );
          Reflect.set(globalThis, 'setTimeout', originalSetTimeout);
        }
      });
    });

    describe('given the browser has no cooperative scheduler available', () => {
      it('completes without yielding while preserving output order', async () => {
        // Arrange
        const network = createWindowTestNetwork();
        const originalWindow = Reflect.get(globalThis, 'window');
        const originalRequestAnimationFrame = Reflect.get(
          globalThis,
          'requestAnimationFrame',
        );
        const originalSetTimeout = Reflect.get(globalThis, 'setTimeout');

        Reflect.set(globalThis, 'window', {});
        Reflect.set(globalThis, 'requestAnimationFrame', undefined);
        Reflect.set(globalThis, 'setTimeout', undefined);

        try {
          // Act
          const outputRows = await forwardWindowedAsync.call(
            network as never,
            [[1], [2], [3]],
            {
              windowSize: 2,
            },
          );

          // Assert
          expect(outputRows).toEqual([[1], [2], [3]]);
        } finally {
          Reflect.set(globalThis, 'window', originalWindow);
          Reflect.set(
            globalThis,
            'requestAnimationFrame',
            originalRequestAnimationFrame,
          );
          Reflect.set(globalThis, 'setTimeout', originalSetTimeout);
        }
      });
    });

    describe('given the runtime resolves as Node', () => {
      it('completes without a default browser yield hook', async () => {
        // Arrange
        const network = createWindowTestNetwork();
        const originalWindow = Reflect.get(globalThis, 'window');

        Reflect.set(globalThis, 'window', undefined);

        try {
          // Act
          const outputRows = await forwardWindowedAsync.call(
            network as never,
            [[1], [2], [3]],
            {
              windowSize: 2,
            },
          );

          // Assert
          expect(outputRows).toEqual([[1], [2], [3]]);
        } finally {
          Reflect.set(globalThis, 'window', originalWindow);
        }
      });
    });
  });
});
