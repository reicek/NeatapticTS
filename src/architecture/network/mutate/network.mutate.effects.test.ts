import { config } from '../../../config';
import { Network, methods } from '../../../neataptic';
import { NetworkMutateMethodRequiredError } from './network.mutate.errors';

const outputChangingMutations = [
  methods.mutation.ADD_NODE,
  methods.mutation.SUB_NODE,
  methods.mutation.ADD_LSTM_NODE,
  methods.mutation.ADD_GRU_NODE,
];

type NeatapticNode = InstanceType<typeof Network>['nodes'][number];

jest.setTimeout(5_000);
jest.retryTimes(3, { logErrorsBeforeRetry: true });

function suppressConsoleNoise(callback: () => void): void {
  const originalWarn = console.warn;
  const originalLog = console.log;
  console.warn = jest.fn();
  console.log = jest.fn();

  try {
    callback();
  } finally {
    console.warn = originalWarn;
    console.log = originalLog;
  }
}

function createSeededRandom(seed: number): () => number {
  let state = seed % 2_147_483_647;

  if (state <= 0) {
    state += 2_147_483_646;
  }

  return () => {
    state = Math.imul(48_271, state) % 2_147_483_647;
    return state / 2_147_483_647;
  };
}

function computeLongestPathDepth(
  startNode: NeatapticNode | undefined,
  targetNode: NeatapticNode | undefined,
  visitedNodes: Set<NeatapticNode> = new Set(),
): number {
  if (startNode == null || targetNode == null) {
    return 0;
  }

  if (startNode === targetNode) {
    return 0;
  }

  visitedNodes.add(startNode);
  let maximumDepth = 0;

  for (const connectionEntry of startNode.connections.out) {
    const destinationNode = connectionEntry.to as NeatapticNode;

    if (visitedNodes.has(destinationNode)) {
      continue;
    }

    const candidateDepth =
      1 + computeLongestPathDepth(destinationNode, targetNode, visitedNodes);

    if (candidateDepth > maximumDepth) {
      maximumDepth = candidateDepth;
    }
  }

  visitedNodes.delete(startNode);
  return maximumDepth;
}

describe('network mutate effects chapter', () => {
  describe('output side effects', () => {
    outputChangingMutations.forEach((mutationMethod) => {
      describe(`given ${mutationMethod.name} runs on a small network`, () => {
        describe('when the same input is activated before and after mutation', () => {
          it('changes the output vector', () => {
            // Arrange
            const nextRandomValue = createSeededRandom(42);
            const network = new Network(2, 1);

            if (mutationMethod === methods.mutation.SUB_NODE) {
              network.mutate(methods.mutation.ADD_NODE);
            }

            const inputValues = [nextRandomValue(), nextRandomValue()];
            const originalOutput = network.activate(inputValues);

            // Act
            network.mutate(mutationMethod);
            const mutatedOutput = network.activate(inputValues);

            // Assert
            expect(mutatedOutput).not.toEqual(originalOutput);
          });
        });
      });
    });
  });

  describe('invalid mutation descriptors', () => {
    describe('given an empty mutation object is provided', () => {
      describe('when mutate() is called', () => {
        it('does not throw', () => {
          // Arrange
          const network = new Network(2, 1);
          const mutateWithEmptyDescriptor = () => {
            suppressConsoleNoise(() => {
              network.mutate({} as unknown as never);
            });
          };

          // Assert
          expect(mutateWithEmptyDescriptor).not.toThrow();
        });
      });
    });

    describe('given the mutation method is null', () => {
      describe('when mutate() is called', () => {
        it('throws the mutate-method-required error', () => {
          // Arrange
          const network = new Network(2, 1);
          const mutateWithNullMethod = () => {
            network.mutate(null as unknown as never);
          };

          // Assert
          expect(mutateWithNullMethod).toThrow(
            NetworkMutateMethodRequiredError,
          );
        });
      });
    });
  });

  describe('deterministic chain growth', () => {
    describe('given repeated ADD_NODE mutations run in deterministic chain mode', () => {
      describe('when the longest path is measured from input to output', () => {
        it('creates a path deeper than two edges', () => {
          // Arrange
          const network = new Network(1, 1);
          const previousDeterministicChainMode = config.deterministicChainMode;

          try {
            config.deterministicChainMode = true;

            for (let mutationIndex = 0; mutationIndex < 4; mutationIndex++) {
              network.mutate(methods.mutation.ADD_NODE);
            }

            const inputNode = network.nodes.find(
              (nodeEntry) => nodeEntry.type === 'input',
            );
            const outputNode = network.nodes.find(
              (nodeEntry) => nodeEntry.type === 'output',
            );

            // Act
            const pathDepth = computeLongestPathDepth(inputNode, outputNode);

            // Assert
            expect(pathDepth).toBeGreaterThan(2);
          } finally {
            config.deterministicChainMode = previousDeterministicChainMode;
          }
        });
      });
    });

    describe('given no structural mutations run', () => {
      describe('when the longest path is measured in a fresh 1-1 network', () => {
        it('stays at a single edge', () => {
          // Arrange
          const network = new Network(1, 1);
          const inputNode = network.nodes.find(
            (nodeEntry) => nodeEntry.type === 'input',
          );
          const outputNode = network.nodes.find(
            (nodeEntry) => nodeEntry.type === 'output',
          );

          // Act
          const pathDepth = computeLongestPathDepth(inputNode, outputNode);

          // Assert
          expect(pathDepth).toBe(1);
        });
      });
    });
  });
});
