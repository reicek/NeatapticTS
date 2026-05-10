import { Architect, Network } from '../../../neataptic';
import Node from '../../node';
import type { NetworkStandaloneProps } from '../network.types';
import {
  NETWORK_STANDALONE_INPUT_SIZE_MISMATCH_ERROR_NAME,
  NetworkStandaloneNoOutputNodesError,
} from './network.standalone.errors';
import { buildNodeSumExpression } from './network.standalone.utils.graph';
import { appendAllNodeComputationLines } from './network.standalone.utils.loop';
import {
  createGenerationContext,
  resolveStandaloneExecutionMetadata,
  seedNodeIndexesAndState,
} from './network.standalone.utils.setup';
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

function calculateMedianAbsoluteDifference(
  runtimeOutput: number[],
  standaloneOutput: number[],
): number {
  if (runtimeOutput.length !== standaloneOutput.length) {
    return Number.POSITIVE_INFINITY;
  }

  const sortedAbsoluteDifferences = runtimeOutput
    .map((runtimeValue, outputIndex) => {
      const standaloneValue = standaloneOutput[outputIndex];
      if (standaloneValue == null) {
        return Number.POSITIVE_INFINITY;
      }

      return Math.abs(runtimeValue - standaloneValue);
    })
    .toSorted((leftDifference, rightDifference) => {
      return leftDifference - rightDifference;
    });

  return sortedAbsoluteDifferences.at(
    Math.floor(sortedAbsoluteDifferences.length / 2),
  ) ?? 0;
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

        it('also honors shared precision config when the raw alias is absent', () => {
          // Arrange
          const network = new Network(2, 1, { seed: 457 });
          const networkWithPrecision = network as unknown as {
            _precisionConfig?: { activationPrecision: 'f32' | 'f64' };
          };

          networkWithPrecision._precisionConfig = {
            activationPrecision: 'f32',
          };

          // Act
          const standaloneSource = network.standalone();

          // Assert
          expect(standaloneSource.includes('new Float32Array([')).toBe(true);
        });

        it('falls back to Float64Array buffers when no precision carrier is set', () => {
          // Arrange
          const network = new Network(2, 1, { seed: 458 });
          const networkWithPrecision = network as unknown as {
            _activationPrecision?: string;
            _precisionConfig?: { activationPrecision: 'f32' | 'f64' };
          };

          networkWithPrecision._activationPrecision = undefined;
          networkWithPrecision._precisionConfig = undefined;

          // Act
          const standaloneSource = network.standalone();

          // Assert
          expect(standaloneSource.includes('new Float64Array([')).toBe(true);
        });
      });
    });

    describe('given float16 activation precision is requested', () => {
      describe('when standalone() is called', () => {
        it('emits Uint16Array activation and state storage buffers', () => {
          // Arrange
          const network = new Network(2, 1, { seed: 459 });
          const networkWithPrecision = network as unknown as {
            _activationPrecision?: string;
            _precisionConfig?: { activationPrecision: 'f16' | 'f32' | 'f64' };
          };

          networkWithPrecision._activationPrecision = undefined;
          networkWithPrecision._precisionConfig = {
            activationPrecision: 'f16',
          };

          // Act
          const standaloneSource = network.standalone();

          // Assert
          expect(
            standaloneSource.includes('var A = new Uint16Array([') &&
              standaloneSource.includes('var S = new Uint16Array(['),
          ).toBe(true);
        });

        it('keeps median output drift below the float16 prototype threshold', () => {
          // Arrange
          const network = new Network(2, 1, { seed: 460 });
          const networkWithPrecision = network as unknown as {
            _activationPrecision?: string;
            _precisionConfig?: { activationPrecision: 'f16' | 'f32' | 'f64' };
          };

          networkWithPrecision._activationPrecision = undefined;
          networkWithPrecision._precisionConfig = {
            activationPrecision: 'f16',
          };
          const standaloneActivate = compileStandaloneActivator(network);
          const sampleInputs = [
            [0.1, 0.2],
            [0.2, 0.1],
            [0.5, 0.25],
            [0.9, 0.1],
            [0.7, 0.6],
            [0.11, 0.44],
            [0.95, 0.15],
            [0.33, 0.66],
          ];

          // Act
          const medianAbsoluteDifference = calculateMedianAbsoluteDifference(
            sampleInputs.flatMap((inputVector) => network.activate(inputVector)),
            sampleInputs.flatMap((inputVector) => standaloneActivate(inputVector)),
          );

          // Assert
          expect(medianAbsoluteDifference < 1e-4).toBe(true);
        });

        it('encodes mantissa rollover seeds into float16 storage literals', () => {
          // Arrange
          const network = new Network(2, 1, { seed: 461 });
          const networkWithPrecision = network as unknown as {
            _activationPrecision?: string;
            _precisionConfig?: { activationPrecision: 'f16' | 'f32' | 'f64' };
          };

          networkWithPrecision._activationPrecision = undefined;
          networkWithPrecision._precisionConfig = {
            activationPrecision: 'f16',
          };
          network.nodes[0].activation = 32760;

          // Act
          const standaloneSource = network.standalone();

          // Assert
          expect(standaloneSource.includes('new Uint16Array([30720')).toBe(
            true,
          );
        });

        it('encodes float16 special values into Uint16 storage literals', () => {
          // Arrange
          const network = new Network(3, 2, { seed: 462 });
          const networkWithPrecision = network as unknown as {
            _activationPrecision?: string;
            _precisionConfig?: { activationPrecision: 'f16' | 'f32' | 'f64' };
          };

          networkWithPrecision._activationPrecision = undefined;
          networkWithPrecision._precisionConfig = {
            activationPrecision: 'f16',
          };
          network.nodes[0].activation = Number.NaN;
          network.nodes[1].activation = Number.POSITIVE_INFINITY;
          network.nodes[2].activation = Number.NEGATIVE_INFINITY;
          network.nodes[3].activation = 70_000;
          network.nodes[4].activation = 5.960464477539063e-8;

          // Act
          const standaloneSource = network.standalone();

          // Assert
          expect(
            standaloneSource.includes('new Uint16Array([32256,31744,64512,31743,1'),
          ).toBe(true);
        });

        it('encodes signed zero and negative finite seeds into Uint16 storage literals', () => {
          // Arrange
          const network = new Network(2, 1, { seed: 467 });
          const networkWithPrecision = network as unknown as {
            _activationPrecision?: string;
            _precisionConfig?: { activationPrecision: 'f16' | 'f32' | 'f64' };
          };

          networkWithPrecision._activationPrecision = undefined;
          networkWithPrecision._precisionConfig = {
            activationPrecision: 'f16',
          };
          network.nodes[0].activation = -0;
          network.nodes[1].activation = -1;

          // Act
          const standaloneSource = network.standalone();

          // Assert
          expect(standaloneSource.includes('new Uint16Array([32768,48128')).toBe(
            true,
          );
        });
      });
    });

    describe('given standalone helper edge cases are exercised directly', () => {
      describe('when execution metadata is resolved without seeded indexes', () => {
        it('skips recomputing topology and drops unindexed traversal entries', () => {
          // Arrange
          const network = new Network(2, 1, { seed: 463 });
          const computeTopoOrder = jest.fn();
          const scheduleAwareNetwork = network as unknown as {
            _computeTopoOrder: () => void;
            _topoDirty?: boolean;
            _topoOrder?: unknown;
          };

          scheduleAwareNetwork._computeTopoOrder = computeTopoOrder;
          scheduleAwareNetwork._topoDirty = false;
          scheduleAwareNetwork._topoOrder = [];
          network.nodes.forEach((node) => {
            const indexedNode = node as typeof node & { index?: number };
            indexedNode.index = undefined;
          });
          const generationContext = createGenerationContext(
            network as unknown as NetworkStandaloneProps,
          );

          // Act
          resolveStandaloneExecutionMetadata(network, generationContext);

          // Assert
          expect({
            computeTopoOrderCalls: computeTopoOrder.mock.calls.length,
            inputIndexes: generationContext.inputNodeIndexes,
            activationIndexes: generationContext.activationNodeIndexes,
            outputIndexes: generationContext.outputNodeIndexes,
          }).toEqual({
            computeTopoOrderCalls: 0,
            inputIndexes: [],
            activationIndexes: [],
            outputIndexes: [],
          });
        });
      });

      describe('when runtime input roles are incomplete', () => {
        it('falls back to node-order input indexes', () => {
          // Arrange
          const network = new Network(2, 1, { seed: 464 });
          const standaloneRoleNetwork = network as unknown as {
            _inputNodeIds?: number[];
          };

          standaloneRoleNetwork._inputNodeIds = [];
          const generationContext = createGenerationContext(
            network as unknown as NetworkStandaloneProps,
          );
          seedNodeIndexesAndState(generationContext);

          // Act
          resolveStandaloneExecutionMetadata(network, generationContext);

          // Assert
          expect(generationContext.inputNodeIndexes).toEqual([0, 1]);
        });
      });

      describe('when a node sum sees only detached incoming nodes', () => {
        it('falls back to the zero-term expression', () => {
          // Arrange
          const network = new Network(1, 1, { seed: 465 });
          const outputNode = network.nodes.find(
            (candidateNode) => candidateNode.type === 'output',
          );

          if (!outputNode) {
            throw new Error('Expected one output node to exist');
          }

          const generationContext = createGenerationContext(
            network as unknown as NetworkStandaloneProps,
          );
          seedNodeIndexesAndState(generationContext);
          const detachedInputNode = {} as Node;
          const indexedOutputNode = outputNode as typeof outputNode & {
            index: number;
          };
          outputNode.connections.in = [
            {
              from: detachedInputNode,
              weight: 1,
              gater: null,
            },
          ] as typeof outputNode.connections.in;
          outputNode.connections.self = [] as typeof outputNode.connections.self;

          // Act
          const sumExpression = buildNodeSumExpression(
            generationContext,
            outputNode,
            indexedOutputNode.index,
          );

          // Assert
          expect(sumExpression).toBe('0');
        });
      });

      describe('when a float16 node sum includes a recurrent self connection', () => {
        it('reads the working state buffer in the generated expression', () => {
          // Arrange
          const network = new Network(1, 1, { seed: 468 });
          const networkWithPrecision = network as unknown as {
            _precisionConfig?: { activationPrecision: 'f16' | 'f32' | 'f64' };
          };
          const outputNode = network.nodes.find(
            (candidateNode) => candidateNode.type === 'output',
          );

          if (!outputNode) {
            throw new Error('Expected one output node to exist');
          }

          networkWithPrecision._precisionConfig = {
            activationPrecision: 'f16',
          };
          const generationContext = createGenerationContext(
            network as unknown as NetworkStandaloneProps,
          );
          seedNodeIndexesAndState(generationContext);
          const indexedOutputNode = outputNode as typeof outputNode & {
            index: number;
          };
          outputNode.connections.self = [
            {
              weight: 1,
              gater: null,
            },
          ] as typeof outputNode.connections.self;

          // Act
          const sumExpression = buildNodeSumExpression(
            generationContext,
            outputNode,
            indexedOutputNode.index,
          );

          // Assert
          expect(sumExpression.includes('WS[')).toBe(true);
        });
      });

      describe('when a generated node uses a non-identity mask', () => {
        it('keeps the multiplicative mask suffix in the emitted body line', () => {
          // Arrange
          const network = new Network(1, 1, { seed: 466 });
          const networkWithPrecision = network as unknown as {
            _precisionConfig?: { activationPrecision: 'f16' | 'f32' | 'f64' };
          };
          const outputNode = network.nodes.find(
            (candidateNode) => candidateNode.type === 'output',
          );

          if (!outputNode) {
            throw new Error('Expected one output node to exist');
          }

          networkWithPrecision._precisionConfig = {
            activationPrecision: 'f16',
          };
          outputNode.mask = 0.5;
          const generationContext = createGenerationContext(
            network as unknown as NetworkStandaloneProps,
          );
          seedNodeIndexesAndState(generationContext);
          resolveStandaloneExecutionMetadata(network, generationContext);

          // Act
          appendAllNodeComputationLines(generationContext);

          // Assert
          expect(
            generationContext.bodyLines.some((bodyLine) =>
              bodyLine.includes('* 0.5'),
            ),
          ).toBe(true);
        });
      });
    });
  });
});
