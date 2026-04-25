import { Architect, Network } from '../../../neataptic';
import {
  NETWORK_STANDALONE_INPUT_SIZE_MISMATCH_ERROR_NAME,
  NetworkStandaloneNoOutputNodesError,
} from './network.standalone.errors';
import { stripCoverage } from './network.standalone.utils.coverage';

type StandaloneActivator = (input: number[] | null | undefined) => number[];

function compileStandaloneActivator(network: Network): StandaloneActivator {
  const standaloneSource = stripCoverage(network.standalone());
  return new Function(`return ${standaloneSource}`)() as StandaloneActivator;
}

function captureThrownError(callback: () => void): Error {
  try {
    callback();
  } catch (error) {
    if (error instanceof Error) {
      return error;
    }

    throw new Error('Expected callback to throw an Error instance', {
      cause: error,
    });
  }

  throw new Error('Expected callback to throw');
}

function createDeterministicInput(inputSize: number): number[] {
  return Array.from(
    { length: inputSize },
    (_unusedValue, inputIndex) => (inputIndex + 1) / 10,
  );
}

function areOutputsClose(
  runtimeOutput: number[],
  standaloneOutput: number[],
  toleranceDigits: number,
): boolean {
  if (runtimeOutput.length !== standaloneOutput.length) {
    return false;
  }

  return runtimeOutput.every((runtimeValue, outputIndex) => {
    const standaloneValue = standaloneOutput[outputIndex];
    if (standaloneValue == null) {
      return false;
    }

    return (
      Math.abs(runtimeValue - standaloneValue) < Math.pow(10, -toleranceDigits)
    );
  });
}

describe('network standalone chapter', () => {
  describe('stripCoverage()', () => {
    describe('given a source string with an Istanbul ignore block', () => {
      describe('when stripCoverage() is called', () => {
        it('removes the ignore directive', () => {
          // Arrange
          const instrumentedSource =
            '/* istanbul ignore next */\nfunction foo() {}';

          // Act
          const cleanedSource = stripCoverage(instrumentedSource);

          // Assert
          expect(cleanedSource.includes('istanbul')).toBe(false);
        });
      });
    });

    describe('given a source string with coverage counters', () => {
      describe('when stripCoverage() is called', () => {
        it('removes the coverage counter calls', () => {
          // Arrange
          const instrumentedSource =
            'cov_123().s[0]++; cov_123().f[1]++; cov_123()';

          // Act
          const cleanedSource = stripCoverage(instrumentedSource);

          // Assert
          expect(cleanedSource.includes('cov_123')).toBe(false);
        });
      });
    });

    describe('given a source string with a source map comment', () => {
      describe('when stripCoverage() is called', () => {
        it('removes the source map marker', () => {
          // Arrange
          const instrumentedSource = '//# sourceMappingURL=foo.js';

          // Act
          const cleanedSource = stripCoverage(instrumentedSource);

          // Assert
          expect(cleanedSource.includes('sourceMappingURL')).toBe(false);
        });
      });
    });

    describe('given a source string with punctuation-only tokens', () => {
      describe('when stripCoverage() is called', () => {
        it('collapses the source down to an empty string', () => {
          // Arrange
          const instrumentedSource = ';;,,';

          // Act
          const cleanedSource = stripCoverage(instrumentedSource);

          // Assert
          expect(cleanedSource.trim()).toBe('');
        });
      });
    });
  });

  describe('Network.standalone()', () => {
    const equivalenceCases = [
      {
        architectureName: 'Perceptron',
        createNetwork: () => Architect.perceptron(2, 3, 1),
      },
      {
        architectureName: 'Basic Network',
        createNetwork: () => new Network(2, 1, { seed: 450 }),
      },
      {
        architectureName: 'LSTM',
        createNetwork: () => Architect.lstm(2, 3, 1),
      },
      {
        architectureName: 'GRU',
        createNetwork: () => Architect.gru(2, 3, 1),
      },
      {
        architectureName: 'Random',
        createNetwork: () => Architect.random(2, 3, 1),
      },
      {
        architectureName: 'RandomSparse',
        createNetwork: () =>
          Architect.randomSparse(2, 3, 1, {
            connections: 6,
            seed: 702,
          }),
      },
      {
        architectureName: 'NARX',
        createNetwork: () => Architect.narx(2, 2, 2, 2, 1),
      },
      {
        architectureName: 'Hopfield',
        createNetwork: () => Architect.hopfield(3),
      },
    ];

    equivalenceCases.forEach(({ architectureName, createNetwork }) => {
      describe(`given a ${architectureName} network`, () => {
        let network: Network;
        let standaloneActivator: StandaloneActivator;
        let inputValues: number[];
        let runtimeOutput: number[];
        let standaloneOutput: number[];

        beforeAll(() => {
          // Arrange
          network = createNetwork();
          standaloneActivator = compileStandaloneActivator(network);
          inputValues = createDeterministicInput(network.input);

          // Act
          runtimeOutput = network.activate(inputValues);
          standaloneOutput = standaloneActivator(inputValues);
        });

        describe('when comparing output shapes', () => {
          it('returns the same output length as the runtime network', () => {
            // Assert
            expect(standaloneOutput.length).toBe(runtimeOutput.length);
          });
        });

        describe('when comparing output values', () => {
          it('keeps the standalone output numerically close to the runtime output', () => {
            // Arrange
            const outputsAreClose = areOutputsClose(
              runtimeOutput,
              standaloneOutput,
              3,
            );

            // Assert
            expect(outputsAreClose).toBe(true);
          });
        });

        describe('when input length is wrong', () => {
          it('throws the standalone input-size mismatch error name', () => {
            // Arrange
            const thrownError = captureThrownError(() => {
              standaloneActivator(inputValues.concat(1));
            });

            // Assert
            expect(thrownError.name).toBe(
              NETWORK_STANDALONE_INPUT_SIZE_MISMATCH_ERROR_NAME,
            );
          });
        });
      });
    });

    describe('given a generated standalone activator', () => {
      const invalidInputCases = [
        {
          architectureName: 'Perceptron',
          createNetwork: () => Architect.perceptron(2, 3, 1),
        },
        {
          architectureName: 'LSTM',
          createNetwork: () => Architect.lstm(2, 4, 1),
        },
        {
          architectureName: 'GRU',
          createNetwork: () => Architect.gru(2, 2, 1),
        },
      ];

      invalidInputCases.forEach(({ architectureName, createNetwork }) => {
        describe(`when ${architectureName} receives null input`, () => {
          it('throws the standalone input-size mismatch error name', () => {
            // Arrange
            const standaloneActivator =
              compileStandaloneActivator(createNetwork());
            const thrownError = captureThrownError(() => {
              standaloneActivator(null);
            });

            // Assert
            expect(thrownError.name).toBe(
              NETWORK_STANDALONE_INPUT_SIZE_MISMATCH_ERROR_NAME,
            );
          });
        });

        describe(`when ${architectureName} receives an empty input array`, () => {
          it('throws the standalone input-size mismatch error name', () => {
            // Arrange
            const standaloneActivator =
              compileStandaloneActivator(createNetwork());
            const thrownError = captureThrownError(() => {
              standaloneActivator([]);
            });

            // Assert
            expect(thrownError.name).toBe(
              NETWORK_STANDALONE_INPUT_SIZE_MISMATCH_ERROR_NAME,
            );
          });
        });

        describe(`when ${architectureName} receives NaN inputs`, () => {
          it('propagates NaN values into the standalone outputs', () => {
            // Arrange
            const network = createNetwork();
            const standaloneActivator = compileStandaloneActivator(network);
            const nanInputValues = Array(network.input).fill(NaN);

            // Act
            const nanOutputValues = standaloneActivator(nanInputValues);

            // Assert
            expect(nanOutputValues.every((value) => Number.isNaN(value))).toBe(
              true,
            );
          });
        });
      });
    });

    describe('given the network has no output nodes', () => {
      describe('when standalone() is called', () => {
        it('throws the no-output-nodes error type', () => {
          // Arrange
          const network = new Network(2, 1, { seed: 451 });
          network.nodes = network.nodes.filter(
            (candidateNode) => candidateNode.type !== 'output',
          );

          // Act
          const createStandaloneSource = () => network.standalone();

          // Assert
          expect(createStandaloneSource).toThrow(
            NetworkStandaloneNoOutputNodesError,
          );
        });
      });
    });

    describe('given the network uses an unknown custom squash function', () => {
      describe('when standalone() is called', () => {
        it('still returns standalone source text', () => {
          // Arrange
          const network = new Network(2, 1, { seed: 452 });
          network.nodes[0].squash = (inputValue: number) => inputValue * 2;

          // Act
          const standaloneSource = network.standalone();

          // Assert
          expect(typeof standaloneSource).toBe('string');
        });
      });
    });

    describe('given the network has no nodes', () => {
      describe('when standalone() is called', () => {
        it('throws the no-output-nodes error type', () => {
          // Arrange
          const network = new Network(2, 1, { seed: 453 });
          network.nodes = [];

          // Act
          const createStandaloneSource = () => network.standalone();

          // Assert
          expect(createStandaloneSource).toThrow(
            NetworkStandaloneNoOutputNodesError,
          );
        });
      });
    });

    describe('given the network only has input nodes', () => {
      describe('when standalone() is called', () => {
        it('throws the no-output-nodes error type', () => {
          // Arrange
          const network = new Network(2, 1, { seed: 454 });
          network.nodes = network.nodes.filter(
            (candidateNode) => candidateNode.type === 'input',
          );

          // Act
          const createStandaloneSource = () => network.standalone();

          // Assert
          expect(createStandaloneSource).toThrow(
            NetworkStandaloneNoOutputNodesError,
          );
        });
      });
    });

    describe('given the network only has hidden nodes', () => {
      describe('when standalone() is called', () => {
        it('throws the no-output-nodes error type', () => {
          // Arrange
          const network = new Network(2, 1, { seed: 455 });
          network.nodes = network.nodes.filter(
            (candidateNode) => candidateNode.type === 'hidden',
          );

          // Act
          const createStandaloneSource = () => network.standalone();

          // Assert
          expect(createStandaloneSource).toThrow(
            NetworkStandaloneNoOutputNodesError,
          );
        });
      });
    });

    describe('given float32 activation precision is requested', () => {
      describe('when standalone() is called', () => {
        it('emits Float32Array state and activation buffers', () => {
          // Arrange
          const network = new Network(2, 1, { seed: 456 });
          const networkWithPrecision = network as unknown as {
            _activationPrecision?: string;
          };
          networkWithPrecision._activationPrecision = 'f32';

          // Act
          const standaloneSource = network.standalone();

          // Assert
          expect(standaloneSource.includes('new Float32Array([')).toBe(true);
        });
      });
    });
  });
});
