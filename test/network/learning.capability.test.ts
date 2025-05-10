import { Architect, Network, methods } from '../../src/neataptic';

function learnSet(
  set: { input: number[]; output: number[] }[],
  iterations: number,
  error: number,
  rate: number = 0.3
): void {
  // Arrange
  const network = Architect.perceptron(
    set[0].input.length,
    5,
    set[0].output.length
  );
  const options = {
    iterations: iterations,
    error: error,
    shuffle: true,
    rate: rate,
    momentum: 0.9,
  };
  // Act
  const results = network.train(set, options);
  // Assert
  expect(results.error).toBeLessThan(error);
}

describe('Learning Capability', () => {
  describe('Logic Gates', () => {
    describe('Scenario: AND gate', () => {
      test('learns the AND gate', () => {
        // Arrange
        const dataset = [
          { input: [0, 0], output: [0] },
          { input: [0, 1], output: [0] },
          { input: [1, 0], output: [0] },
          { input: [1, 1], output: [1] },
        ];
        const network = Architect.perceptron(2, 5, 1);
        const options = { iterations: 1000, error: 0.002, shuffle: true, rate: 0.3, momentum: 0.9 };
        // Act
        const results = network.train(dataset, options);
        // Assert
        expect(results.error).toBeLessThan(0.002);
      });
      test('fails to learn AND with insufficient iterations', () => {
        // Arrange
        const dataset = [
          { input: [0, 0], output: [0] },
          { input: [0, 1], output: [0] },
          { input: [1, 0], output: [0] },
          { input: [1, 1], output: [1] },
        ];
        const network = Architect.perceptron(2, 5, 1);
        const options = { iterations: 1, error: 0.002, shuffle: true, rate: 0.3, momentum: 0.9 };
        // Act
        const results = network.train(dataset, options);
        // Assert
        expect(results.error).toBeGreaterThanOrEqual(0.002);
      });
    });

    describe('Scenario: XOR gate', () => {
      test('learns the XOR gate', () => {
        // Arrange
        const dataset = [
          { input: [0, 0], output: [0] },
          { input: [0, 1], output: [1] },
          { input: [1, 0], output: [1] },
          { input: [1, 1], output: [0] },
        ];
        const network = Architect.perceptron(2, 5, 1);
        const options = { iterations: 10000, error: 0.03, shuffle: true, rate: 0.1, momentum: 0.9 };
        // Act
        const results = network.train(dataset, options);
        // Assert
        expect(results.error).toBeLessThan(0.03);
      });
      test('fails to learn XOR with insufficient iterations', () => {
        // Arrange
        const dataset = [
          { input: [0, 0], output: [0] },
          { input: [0, 1], output: [1] },
          { input: [1, 0], output: [1] },
          { input: [1, 1], output: [0] },
        ];
        const network = Architect.perceptron(2, 5, 1);
        const options = { iterations: 1, error: 0.03, shuffle: true, rate: 0.1, momentum: 0.9 };
        // Act
        const results = network.train(dataset, options);
        // Assert
        expect(results.error).toBeGreaterThanOrEqual(0.03);
      });
    });

    describe('Scenario: NOT gate', () => {
      test('learns the NOT gate', () => {
        // Arrange
        const dataset = [
          { input: [0], output: [1] },
          { input: [1], output: [0] },
        ];
        const network = Architect.perceptron(1, 5, 1);
        const options = { iterations: 1000, error: 0.002, shuffle: true, rate: 0.3, momentum: 0.9 };
        // Act
        const results = network.train(dataset, options);
        // Assert
        expect(results.error).toBeLessThan(0.002);
      });
      test('fails to learn NOT with insufficient iterations', () => {
        // Arrange
        const dataset = [
          { input: [0], output: [1] },
          { input: [1], output: [0] },
        ];
        const network = Architect.perceptron(1, 5, 1);
        const options = { iterations: 1, error: 0.002, shuffle: true, rate: 0.3, momentum: 0.9 };
        // Act
        const results = network.train(dataset, options);
        // Assert
        expect(results.error).toBeGreaterThanOrEqual(0.002);
      });
    });

    describe('Scenario: XNOR gate', () => {
      test('learns the XNOR gate', () => {
        // Arrange
        const dataset = [
          { input: [0, 0], output: [1] },
          { input: [0, 1], output: [0] },
          { input: [1, 0], output: [0] },
          { input: [1, 1], output: [1] },
        ];
        const network = Architect.perceptron(2, 10, 1);
        const options = { iterations: 20000, error: 0.05, shuffle: true, rate: 0.3, momentum: 0.9 };
        // Act
        const results = network.train(dataset, options);
        // Assert
        expect(results.error).toBeLessThanOrEqual(0.05);
      });
      test('fails to learn XNOR with insufficient iterations', () => {
        // Arrange
        const dataset = [
          { input: [0, 0], output: [1] },
          { input: [0, 1], output: [0] },
          { input: [1, 0], output: [0] },
          { input: [1, 1], output: [1] },
        ];
        const network = Architect.perceptron(2, 10, 1);
        const options = { iterations: 1, error: 0.05, shuffle: true, rate: 0.3, momentum: 0.9 };
        // Act
        const results = network.train(dataset, options);
        // Assert
        expect(results.error).toBeGreaterThanOrEqual(0.05);
      });
    });

    describe('Scenario: OR gate', () => {
      test('learns the OR gate', () => {
        // Arrange
        const dataset = [
          { input: [0, 0], output: [0] },
          { input: [0, 1], output: [1] },
          { input: [1, 0], output: [1] },
          { input: [1, 1], output: [1] },
        ];
        const network = Architect.perceptron(2, 5, 1);
        const options = { iterations: 1000, error: 0.002, shuffle: true, rate: 0.3, momentum: 0.9 };
        // Act
        const results = network.train(dataset, options);
        // Assert
        expect(results.error).toBeLessThan(0.002);
      });
      test('fails to learn OR with insufficient iterations', () => {
        // Arrange
        const dataset = [
          { input: [0, 0], output: [0] },
          { input: [0, 1], output: [1] },
          { input: [1, 0], output: [1] },
          { input: [1, 1], output: [1] },
        ];
        const network = Architect.perceptron(2, 5, 1);
        const options = { iterations: 1, error: 0.002, shuffle: true, rate: 0.3, momentum: 0.9 };
        // Act
        const results = network.train(dataset, options);
        // Assert
        expect(results.error).toBeGreaterThanOrEqual(0.002);
      });
    });
  });

  describe('Function Approximation', () => {
    describe('Scenario: SIN function', () => {
      test('learns the SIN function', () => {
        jest.setTimeout(30000);
        // Arrange
        const set: { input: number[]; output: number[] }[] = [];
        while (set.length < 100) {
          const inputValue = Math.random() * Math.PI * 2;
          set.push({
            input: [inputValue / (Math.PI * 2)],
            output: [(Math.sin(inputValue) + 1) / 2],
          });
        }
        const network = Architect.perceptron(1, 5, 1);
        const options = { iterations: 1000, error: 0.05, shuffle: true, rate: 0.3, momentum: 0.9 };
        // Act
        const results = network.train(set, options);
        // Assert
        expect(results.error).toBeLessThan(0.05);
      });
      test('fails to learn SIN with insufficient iterations', () => {
        jest.setTimeout(30000);
        // Arrange
        const set: { input: number[]; output: number[] }[] = [];
        while (set.length < 100) {
          const inputValue = Math.random() * Math.PI * 2;
          set.push({
            input: [inputValue / (Math.PI * 2)],
            output: [(Math.sin(inputValue) + 1) / 2],
          });
        }
        const network = Architect.perceptron(1, 5, 1);
        const options = { iterations: 1, error: 0.05, shuffle: true, rate: 0.3, momentum: 0.9 };
        // Act
        const results = network.train(set, options);
        // Assert
        expect(results.error).toBeGreaterThanOrEqual(0.05);
      });
    });

    describe('Scenario: SIN and COS functions simultaneously', () => {
      test('learns the SIN and COS functions simultaneously', () => {
        jest.setTimeout(30000);
        // Arrange
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
        const network = Architect.perceptron(1, 5, 2);
        const options = { iterations: 5000, error: 0.05, shuffle: true, rate: 0.3, momentum: 0.9 };
        // Act
        const results = network.train(set, options);
        // Assert
        expect(results.error).toBeLessThan(0.05);
      });
      test('fails to learn SIN and COS with insufficient iterations', () => {
        jest.setTimeout(30000);
        // Arrange
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
        const network = Architect.perceptron(1, 5, 2);
        const options = { iterations: 1, error: 0.05, shuffle: true, rate: 0.3, momentum: 0.9 };
        // Act
        const results = network.train(set, options);
        // Assert
        expect(results.error).toBeGreaterThanOrEqual(0.05);
      });
    });
  });

  describe('Comparison & Simple Sequence Tasks', () => {
    test('learns the "Bigger than" comparison', () => {
      jest.setTimeout(30000);
      // Arrange
      const set: { input: number[]; output: number[] }[] = [];
      for (let i = 0; i < 100; i++) {
        const x = Math.random();
        const y = Math.random();
        const z = x > y ? 1 : 0;
        set.push({ input: [x, y], output: [z] });
      }
      // Act & Assert
      learnSet(set, 500, 0.05);
    });
    test('learns the SHIFT sequence task', () => {
      // Arrange
      const set: { input: number[]; output: number[] }[] = [];
      for (let i = 0; i < 1000; i++) {
        const x = Math.random();
        const y = Math.random();
        const z = Math.random();
        set.push({ input: [x, y, z], output: [z, x, y] });
      }
      // Act & Assert
      learnSet(set, 5000, 0.03);
    });
  });

  describe('Recurrent Network Tasks', () => {
    describe('LSTM XOR Sequence', () => {
      jest.setTimeout(120000);
      let lstm: Network;
      beforeAll(() => {
        // Arrange
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
          // Act
          output = lstm.activate([1])[0];
        });
        test('outputs value greater than 0.48', () => {
          // Assert
          expect(output).toBeGreaterThan(0.48);
        });
        test('outputs value less than 1.0', () => {
          // Assert
          expect(output).toBeLessThan(1.0);
        });
      });
      describe('After second 1 input (1 XOR 1)', () => {
        let output: number;
        beforeAll(() => {
          // Act
          output = lstm.activate([1])[0];
        });
        test('outputs value less than 0.52', () => {
          // Assert
          expect(output).toBeLessThan(0.52);
        });
        test('outputs value greater than 0.0', () => {
          // Assert
          expect(output).toBeGreaterThan(0.0);
        });
      });
      describe('After first 0 input (1 XOR 0)', () => {
        let output: number;
        beforeAll(() => {
          // Act
          output = lstm.activate([0])[0];
        });
        test('outputs value greater than 0.48', () => {
          // Assert
          expect(output).toBeGreaterThan(0.48);
        });
        test('outputs value less than 1.0', () => {
          // Assert
          expect(output).toBeLessThan(1.0);
        });
      });
      describe('After second 0 input (0 XOR 0)', () => {
        let output: number;
        beforeAll(() => {
          // Act
          output = lstm.activate([0])[0];
        });
        test('outputs value less than 0.52', () => {
          // Assert
          expect(output).toBeLessThan(0.52);
        });
        test('outputs value greater than 0.0', () => {
          // Assert
          expect(output).toBeGreaterThan(0.0);
        });
      });
    });
    describe('GRU XOR Sequence', () => {
      jest.setTimeout(120000);
      let gru: Network;
      beforeAll(() => {
        // Arrange
        gru = Architect.gru(1, 5, 1);
        gru.train(
          [
            { input: [0], output: [0] },
            { input: [1], output: [1] },
            { input: [1], output: [0] },
            { input: [0], output: [1] },
            { input: [0], output: [0] },
          ],
          { error: 0.005, iterations: 120000, rate: 0.05, clear: true }
        );
        gru.activate([0]);
      });
      describe('After first 1 input (0 XOR 1)', () => {
        let output: number;
        beforeAll(() => {
          // Act
          output = gru.activate([1])[0];
        });
        test('outputs value greater than 0.32', () => {
          // Assert
          expect(output).toBeGreaterThan(0.32);
        });
        test('outputs value less than 1.0', () => {
          // Assert
          expect(output).toBeLessThan(1.0);
        });
      });
      describe('After second 1 input (1 XOR 1)', () => {
        let output: number;
        beforeAll(() => {
          // Act
          output = gru.activate([1])[0];
        });
        test('outputs value less than 0.66', () => {
          // Assert
          expect(output).toBeLessThan(0.66);
        });
        test('outputs value greater than 0.0', () => {
          // Assert
          expect(output).toBeGreaterThan(0.0);
        });
      });
      describe('After first 0 input (1 XOR 0)', () => {
        let output: number;
        beforeAll(() => {
          // Act
          output = gru.activate([0])[0];
        });
        test('outputs value greater than 0.32', () => {
          // Assert
          expect(output).toBeGreaterThan(0.32);
        });
        test('outputs value less than 1.0', () => {
          // Assert
          expect(output).toBeLessThan(1.0);
        });
      });
      describe('After second 0 input (0 XOR 0)', () => {
        let output: number;
        beforeAll(() => {
          // Act
          output = gru.activate([0])[0];
        });
        test('outputs value less than 0.62', () => {
          // Assert
          expect(output).toBeLessThan(0.62);
        });
        test('outputs value greater than 0.0', () => {
          // Assert
          expect(output).toBeGreaterThan(0.0);
        });
      });
    });
    test('NARX learns a specific sequence', () => {
      // Arrange
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
      const options = { iterations: 12000, error: 0.01 };
      // Act
      narx.train(trainingData, options);
      const testResult = narx.test(trainingData);
      // Assert
      expect(testResult.error).toBeLessThan(0.25);
    });
  });
});
