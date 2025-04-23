import { Architect, Network, methods } from '../src/neataptic';

// Retry failed tests
jest.retryTimes(10, { logErrorsBeforeRetry: true });

/**
 * Helper function to verify that a given mutation method alters the network's output.
 * It creates a standard perceptron, records its output for a grid of inputs,
 * applies the specified mutation, records the new output, and asserts that the outputs differ.
 * @param method - The mutation method to test (e.g., methods.mutation.ADD_NODE).
 */
function checkMutation(method: any): void {
  // Arrange: Create a multi-layer perceptron network and apply initial mutations.
  const network = Architect.perceptron(2, 4, 4, 4, 2);
  network.mutate(methods.mutation.ADD_GATE);
  network.mutate(methods.mutation.ADD_BACK_CONN);
  network.mutate(methods.mutation.ADD_SELF_CONN);

  // Arrange: Store the network's output for a range of input values before mutation.
  const originalOutput: number[][] = [];
  let i: number, j: number;
  for (i = 0; i <= 10; i++) {
    for (j = 0; j <= 10; j++) {
      originalOutput.push(network.activate([i / 10, j / 10]));
    }
  }

  // Act: Apply the mutation method being tested.
  network.mutate(method);

  // Act: Store the network's output after the mutation.
  const mutatedOutput: number[][] = [];
  for (i = 0; i <= 10; i++) {
    for (j = 0; j <= 10; j++) {
      mutatedOutput.push(network.activate([i / 10, j / 10]));
    }
  }

  // Assert: Ensure the mutation changed the network's behavior.
  expect(originalOutput).not.toEqual(mutatedOutput);
}

/**
 * Helper function to test if a network can learn a given dataset to a specified error threshold.
 * It creates a perceptron network sized for the dataset, trains it, and asserts
 * that the final error is below the target.
 * @param set - The training dataset, an array of objects with 'input' and 'output' arrays.
 * @param iterations - The maximum number of training iterations.
 * @param error - The target error threshold to achieve.
 * @param rate - The learning rate for training (defaults to 0.3).
 */
function learnSet(
  set: { input: number[]; output: number[] }[],
  iterations: number,
  error: number,
  rate: number = 0.3 // Default learning rate if not provided.
): void {
  // Arrange: Create a perceptron network dynamically sized based on the dataset.
  const network = Architect.perceptron(
    set[0].input.length, // Input layer size matches dataset input length.
    5, // Fixed hidden layer size.
    set[0].output.length // Output layer size matches dataset output length.
  );

  // Arrange: Define training options.
  const options = {
    iterations: iterations, // Maximum training cycles.
    error: error, // Target error level.
    shuffle: true, // Shuffle the dataset before each iteration.
    rate: rate, // Learning rate.
    momentum: 0.9, // Momentum for faster convergence.
  };

  // Act: Train the network.
  const results = network.train(set, options);

  // Assert: Check if the training achieved the desired error threshold.
  expect(results.error).toBeLessThan(error);
}

// Helper function to strip coverage code - remains the same
const stripCoverage = (code: string): string => {
  // Remove Istanbul ignore comments (multi-line).
  code = code.replace(/\/\*\s*istanbul\s+ignore\s+[\s\S]*?\*\//g, '');
  // Remove coverage counter increments (e.g., cov_...). No trailing semicolon assumed.
  code = code.replace(/cov_[\w$]+\(\)\.(s|f|b)\[\d+\](\[\d+\])?\+\+/g, '');
  // Remove simple coverage function calls (e.g., cov_...). No trailing semicolon assumed.
  code = code.replace(/cov_[\w$]+\(\)/g, '');
  // Remove sourceMappingURL comments.
  code = code.replace(/^\s*\/\/# sourceMappingURL=.*\s*$/gm, '');
  // Clean up potential leftover commas from removed code within parentheses.
  code = code.replace(/\(\s*,\s*/g, '( ');
  code = code.replace(/\s*,\s*\)/g, ' )');
  // Trim whitespace.
  code = code.trim();
  // Remove empty statements or lines with only semicolons/commas.
  code = code.replace(/^\s*;\s*$/gm, '');
  code = code.replace(/;{2,}/g, ';');
  code = code.replace(/^\s*[,;]?\s*$/gm, '');
  return code;
};

// Main test suite for Network functionalities.
describe('Network', () => {
  // Test suite for various network mutation methods.
  describe('Mutation Effects', () => {
    // Dynamically create tests for each mutation method
    Object.values(methods.mutation).forEach((method) => {
      if (typeof method === 'function' && method.name) {
        test(`should alter network output when mutating with ${method.name}`, () => {
          checkMutation(method);
        });
      } else if (
        typeof method === 'object' &&
        method !== null &&
        'name' in method
      ) {
        test(`should alter network output when mutating with ${method.name}`, () => {
          checkMutation(method);
        });
      }
    });
  });

  // Test suite for network structure properties and serialization.
  describe('Structure & Serialization', () => {
    describe('Feed-forward Property', () => {
      test('should maintain feed-forward connections after mutations and crossover', () => {
        jest.setTimeout(30000);
        const network1 = new Network(2, 2);
        const network2 = new Network(2, 2);

        let i;
        for (i = 0; i < 100; i++) {
          network1.mutate(methods.mutation.ADD_NODE);
          network2.mutate(methods.mutation.ADD_NODE);
        }
        for (i = 0; i < 400; i++) {
          network1.mutate(methods.mutation.ADD_CONN);
          network2.mutate(methods.mutation.ADD_NODE);
        }

        const network = Network.crossOver(network1, network2);

        const allFeedForward = network.connections.every((conn) => {
          const fromNode = conn.from;
          const toNode = conn.to;
          if (
            network.nodes.includes(fromNode) &&
            network.nodes.includes(toNode)
          ) {
            const fromIndex = network.nodes.indexOf(fromNode);
            const toIndex = network.nodes.indexOf(toNode);
            return fromIndex < toIndex;
          } else {
            console.error(
              `Connection node not found in network nodes array: from=${fromNode?.index}, to=${toNode?.index}`
            );
            return false;
          }
        });

        expect(allFeedForward).toBe(true);
      });
    });

    describe('fromJSON / toJSON Equivalency', () => {
      jest.setTimeout(15000);

      const runEquivalencyTests = (
        architectureName: string,
        createNetwork: () => Network
      ) => {
        describe(`${architectureName}`, () => {
          let original: Network;
          let copy: Network;
          let input: number[];
          let originalOutput: number[];
          let copyOutput: number[];

          beforeAll(() => {
            original = createNetwork();
            const json: any = original.toJSON();
            copy = Network.fromJSON(json);

            input = Array.from({ length: original.input }, () => Math.random());

            originalOutput = original.activate(input);
            copyOutput = copy.activate(input);
          });

          test('should produce the same output length as the original', () => {
            expect(copyOutput.length).toEqual(originalOutput.length);
          });

          test('should produce numerically close outputs to the original', () => {
            expect(copyOutput.length).toEqual(originalOutput.length);
            const outputsAreEqual = copyOutput.every(
              (val, i) =>
                typeof originalOutput[i] === 'number' &&
                Math.abs(val - originalOutput[i]) < 1e-9
            );
            expect(outputsAreEqual).toBe(true);
          });
        });
      };

      runEquivalencyTests('Perceptron', () =>
        Architect.perceptron(
          Math.floor(Math.random() * 5 + 1),
          Math.floor(Math.random() * 5 + 1),
          Math.floor(Math.random() * 5 + 1)
        )
      );
      runEquivalencyTests(
        'Basic Network',
        () =>
          new Network(
            Math.floor(Math.random() * 5 + 1),
            Math.floor(Math.random() * 5 + 1)
          )
      );
      runEquivalencyTests('LSTM', () =>
        Architect.lstm(
          Math.floor(Math.random() * 5 + 1),
          Math.floor(Math.random() * 5 + 1),
          Math.floor(Math.random() * 5 + 1)
        )
      );
      runEquivalencyTests('GRU', () =>
        Architect.gru(
          Math.floor(Math.random() * 5 + 1),
          Math.floor(Math.random() * 5 + 1),
          Math.floor(Math.random() * 5 + 1),
          Math.floor(Math.random() * 5 + 1)
        )
      );
      runEquivalencyTests('Random', () =>
        Architect.random(
          Math.floor(Math.random() * 5 + 1),
          Math.floor(Math.random() * 10 + 1),
          Math.floor(Math.random() * 5 + 1)
        )
      );
      runEquivalencyTests('NARX', () =>
        Architect.narx(
          Math.floor(Math.random() * 5 + 1),
          Math.floor(Math.random() * 5 + 1),
          Math.floor(Math.random() * 5 + 1),
          Math.floor(Math.random() * 5 + 1),
          Math.floor(Math.random() * 5 + 1)
        )
      );
      runEquivalencyTests('Hopfield', () =>
        Architect.hopfield(Math.floor(Math.random() * 5 + 1))
      );
    });

    describe('Standalone Function Equivalency', () => {
      jest.setTimeout(15000);

      const runStandaloneTests = (
        architectureName: string,
        createNetwork: () => Network
      ) => {
        describe(`${architectureName}`, () => {
          let original: Network;
          let standaloneFn: (input: number[]) => number[];
          let input: number[];
          let originalOutput: number[];
          let standaloneOutput: number[];
          let strippedCode: string = '';

          beforeAll(() => {
            try {
              original = createNetwork();
              const standaloneCode = original.standalone();
              strippedCode = stripCoverage(standaloneCode);
              standaloneFn = new Function(`return ${strippedCode}`)();

              input = Array.from({ length: original.input }, () =>
                Math.random()
              );

              originalOutput = original.activate(input);
              standaloneOutput = standaloneFn(input);
            } catch (e) {
              console.error(
                `Error during standalone setup for ${architectureName}. Code:\n`,
                strippedCode
              );
              throw e;
            }
          });

          test('should produce the same output length as the original', () => {
            expect(standaloneOutput.length).toEqual(originalOutput.length);
          });

          test('should produce numerically close outputs to the original', () => {
            expect(standaloneOutput.length).toEqual(originalOutput.length);
            const outputsAreEqual = standaloneOutput.every(
              (val, i) =>
                typeof originalOutput[i] === 'number' &&
                Math.abs(val - originalOutput[i]) < 1e-9
            );
            expect(outputsAreEqual).toBe(true);
          });
        });
      };

      runStandaloneTests('Perceptron', () =>
        Architect.perceptron(
          Math.floor(Math.random() * 2 + 1),
          Math.floor(Math.random() * 3 + 1),
          Math.floor(Math.random() * 2 + 1)
        )
      );
      runStandaloneTests(
        'Basic Network',
        () =>
          new Network(
            Math.floor(Math.random() * 2 + 1),
            Math.floor(Math.random() * 2 + 1)
          )
      );
      runStandaloneTests('LSTM', () =>
        Architect.lstm(
          Math.floor(Math.random() * 2 + 1),
          Math.floor(Math.random() * 3 + 1),
          Math.floor(Math.random() * 2 + 1)
        )
      );
      runStandaloneTests('GRU', () =>
        Architect.gru(
          Math.floor(Math.random() * 2 + 1),
          Math.floor(Math.random() * 2 + 1),
          Math.floor(Math.random() * 2 + 1)
        )
      );
      runStandaloneTests('Random', () =>
        Architect.random(
          Math.floor(Math.random() * 2 + 1),
          Math.floor(Math.random() * 4 + 1),
          Math.floor(Math.random() * 2 + 1)
        )
      );
      runStandaloneTests('NARX', () =>
        Architect.narx(
          Math.floor(Math.random() * 2 + 1),
          Math.floor(Math.random() * 2 + 1),
          Math.floor(Math.random() * 2 + 1),
          Math.floor(Math.random() * 2 + 1),
          Math.floor(Math.random() * 2 + 1)
        )
      );
      runStandaloneTests('Hopfield', () =>
        Architect.hopfield(Math.floor(Math.random() * 3 + 1))
      );
    });
  });

  // Test suite verifying the network's ability to learn various tasks.
  describe('Learning Capability', () => {
    describe('Logic Gates', () => {
      test('should learn the AND gate', () => {
        const dataset = [
          { input: [0, 0], output: [0] },
          { input: [0, 1], output: [0] },
          { input: [1, 0], output: [0] },
          { input: [1, 1], output: [1] },
        ];
        learnSet(dataset, 1000, 0.002);
      });

      test('should learn the XOR gate', () => {
        const dataset = [
          { input: [0, 0], output: [0] },
          { input: [0, 1], output: [1] },
          { input: [1, 0], output: [1] },
          { input: [1, 1], output: [0] },
        ];
        learnSet(dataset, 10000, 0.03, 0.1);
      });

      test('should learn the NOT gate', () => {
        const dataset = [
          { input: [0], output: [1] },
          { input: [1], output: [0] },
        ];
        learnSet(dataset, 1000, 0.002);
      });

      test('should learn the XNOR gate', () => {
        const dataset = [
          { input: [0, 0], output: [1] },
          { input: [0, 1], output: [0] },
          { input: [1, 0], output: [0] },
          { input: [1, 1], output: [1] },
        ];
        learnSet(dataset, 3000, 0.002);
      });

      test('should learn the OR gate', () => {
        const dataset = [
          { input: [0, 0], output: [0] },
          { input: [0, 1], output: [1] },
          { input: [1, 0], output: [1] },
          { input: [1, 1], output: [1] },
        ];
        learnSet(dataset, 1000, 0.002);
      });
    });

    describe('Function Approximation', () => {
      test('should learn the SIN function', () => {
        jest.setTimeout(30000);
        const set: { input: number[]; output: number[] }[] = [];
        while (set.length < 100) {
          const inputValue = Math.random() * Math.PI * 2;
          set.push({
            input: [inputValue / (Math.PI * 2)],
            output: [(Math.sin(inputValue) + 1) / 2],
          });
        }
        learnSet(set, 1000, 0.05);
      });

      test('should learn the SIN and COS functions simultaneously', () => {
        jest.setTimeout(30000);
        const set: { input: number[]; output: number[] }[] = [];
        while (set.length < 100) {
          const inputValue = Math.random() * Math.PI * 2;
          set.push({
            input: [inputValue / (Math.PI * 2)],
            output: [
              (Math.sin(inputValue) + 1) / 2,
              (Math.cos(inputValue) + 1) / 2,
            ],
          });
        }
        learnSet(set, 5000, 0.05);
      });
    });

    describe('Comparison & Simple Sequence Tasks', () => {
      test('should learn the "Bigger than" comparison', () => {
        jest.setTimeout(30000);
        const set: { input: number[]; output: number[] }[] = [];
        for (let i = 0; i < 100; i++) {
          const x = Math.random();
          const y = Math.random();
          const z = x > y ? 1 : 0;
          set.push({ input: [x, y], output: [z] });
        }
        learnSet(set, 500, 0.05);
      });

      test('should learn the SHIFT sequence task', () => {
        const set: { input: number[]; output: number[] }[] = [];
        for (let i = 0; i < 1000; i++) {
          const x = Math.random();
          const y = Math.random();
          const z = Math.random();
          set.push({ input: [x, y, z], output: [z, x, y] });
        }
        learnSet(set, 5000, 0.03);
      });
    });

    describe('Recurrent Network Tasks', () => {
      describe('LSTM XOR Sequence', () => {
        jest.setTimeout(120000);
        let lstm: Network;

        beforeAll(() => {
          lstm = Architect.lstm(1, 15, 1);
          lstm.train(
            [
              { input: [0], output: [0] },
              { input: [1], output: [1] },
              { input: [1], output: [0] },
              { input: [0], output: [1] },
              { input: [0], output: [0] },
            ],
            {
              error: 0.03,
              iterations: 50000,
              rate: 0.05,
              momentum: 0.9,
              clear: true,
            }
          );
          lstm.activate([0]);
        });

        describe('After first 1 input (0 XOR 1)', () => {
          let output: number;
          beforeAll(() => {
            output = lstm.activate([1])[0];
          });
          test('should output value greater than 0.48', () => {
            expect(output).toBeGreaterThan(0.48);
          });
          test('should output value less than 1.0', () => {
            expect(output).toBeLessThan(1.0);
          });
        });

        describe('After second 1 input (1 XOR 1)', () => {
          let output: number;
          beforeAll(() => {
            output = lstm.activate([1])[0];
          });
          test('should output value less than 0.52', () => {
            expect(output).toBeLessThan(0.52);
          });
          test('should output value greater than 0.0', () => {
            expect(output).toBeGreaterThan(0.0);
          });
        });

        describe('After first 0 input (1 XOR 0)', () => {
          let output: number;
          beforeAll(() => {
            output = lstm.activate([0])[0];
          });
          test('should output value greater than 0.48', () => {
            expect(output).toBeGreaterThan(0.48);
          });
          test('should output value less than 1.0', () => {
            expect(output).toBeLessThan(1.0);
          });
        });

        describe('After second 0 input (0 XOR 0)', () => {
          let output: number;
          beforeAll(() => {
            output = lstm.activate([0])[0];
          });
          test('should output value less than 0.52', () => {
            expect(output).toBeLessThan(0.52);
          });
          test('should output value greater than 0.0', () => {
            expect(output).toBeGreaterThan(0.0);
          });
        });
      });

      describe('GRU XOR Sequence', () => {
        jest.setTimeout(60000);
        let gru: Network;

        beforeAll(() => {
          gru = Architect.gru(1, 5, 1);
          gru.train(
            [
              { input: [0], output: [0] },
              { input: [1], output: [1] },
              { input: [1], output: [0] },
              { input: [0], output: [1] },
              { input: [0], output: [0] },
            ],
            { error: 0.005, iterations: 60000, rate: 0.05, clear: true }
          );
          gru.activate([0]);
        });

        describe('After first 1 input (0 XOR 1)', () => {
          let output: number;
          beforeAll(() => {
            output = gru.activate([1])[0];
          });
          test('should output value greater than 0.48', () => {
            expect(output).toBeGreaterThan(0.48);
          });
          test('should output value less than 1.0', () => {
            expect(output).toBeLessThan(1.0);
          });
        });

        describe('After second 1 input (1 XOR 1)', () => {
          let output: number;
          beforeAll(() => {
            output = gru.activate([1])[0];
          });
          test('should output value less than 0.52', () => {
            expect(output).toBeLessThan(0.52);
          });
          test('should output value greater than 0.0', () => {
            expect(output).toBeGreaterThan(0.0);
          });
        });

        describe('After first 0 input (1 XOR 0)', () => {
          let output: number;
          beforeAll(() => {
            output = gru.activate([0])[0];
          });
          test('should output value greater than 0.48', () => {
            expect(output).toBeGreaterThan(0.48);
          });
          test('should output value less than 1.0', () => {
            expect(output).toBeLessThan(1.0);
          });
        });

        describe('After second 0 input (0 XOR 0)', () => {
          let output: number;
          beforeAll(() => {
            output = gru.activate([0])[0];
          });
          test('should output value less than 0.52', () => {
            expect(output).toBeLessThan(0.52);
          });
          test('should output value greater than 0.0', () => {
            expect(output).toBeGreaterThan(0.0);
          });
        });
      });

      test('NARX should learn a specific sequence', () => {
        const narx = Architect.narx(1, 5, 1, 3, 3);
        const trainingData = [
          { input: [0], output: [0] },
          { input: [0], output: [0] },
          { input: [0], output: [1] },
          { input: [1], output: [0] },
          { input: [0], output: [0] },
          { input: [0], output: [0] },
          { input: [0], output: [1] },
        ];
        const options = { iterations: 5000, error: 0.005, rate: 0.05 };

        narx.train(trainingData, options);
        const testResult = narx.test(trainingData);

        expect(testResult.error).toBeLessThan(0.21);
      });
    });
  });
});
