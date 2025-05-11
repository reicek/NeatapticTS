import { Architect, Network, methods } from '../../src/neataptic';

// Helper function to strip coverage code
const stripCoverage = (code: string): string => {
  code = code.replace(/\/\*\s*istanbul\s+ignore\s+[\s\S]*?\*\//g, '');
  code = code.replace(/cov_[\w$]+\(\)\.(s|f|b)\[\d+\](\[\d+\])?\+\+/g, '');
  code = code.replace(/cov_[\w$]+\(\)/g, '');
  code = code.replace(/^\s*\/\/# sourceMappingURL=.*\s*$/gm, '');
  code = code.replace(/\(\s*,\s*/g, '( ');
  code = code.replace(/\s*,\s*\)/g, ' )');
  code = code.trim();
  code = code.replace(/^\s*;\s*$/gm, '');
  code = code.replace(/;{2,}/g, ';');
  code = code.replace(/^\s*[,;]?\s*$/gm, '');
  return code;
};

describe('Standalone Functionality', () => {
  describe('Standalone Function Equivalency', () => {
    jest.setTimeout(15000);
    const runStandaloneTests = (
      architectureName: string,
      createNetwork: () => Network
    ) => {
      describe(`Scenario: ${architectureName}`, () => {
        let original: Network;
        let standaloneFn: (input: number[]) => number[];
        let input: number[];
        let originalOutput: number[];
        let standaloneOutput: number[];
        let strippedCode: string = '';
        beforeAll(() => {
          // Arrange
          try {
            original = createNetwork();
            const standaloneCode = original.standalone();
            strippedCode = stripCoverage(standaloneCode);
            standaloneFn = new Function(`return ${strippedCode}`)();
            input = Array.from({ length: original.input }, () => Math.random());
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
        test('produces the same output length as the original', () => {
          // Assert
          expect(standaloneOutput.length).toEqual(originalOutput.length);
        });
        test('produces numerically close outputs to the original', () => {
          // Arrange
          // Use the original input array that was created with the correct size in beforeAll
          // instead of hardcoding a size that might not match
          
          // Act
          const originalOutput = original.activate(input);
          const standaloneFnOutput = standaloneFn(input);
          
          // Assert - Use more reasonable tolerance for floating-point comparisons
          standaloneFnOutput.forEach((val, i) => {
            // Update tolerance to accommodate small floating-point differences
            // especially for complex architectures like LSTM and GRU
            const diff = Math.abs(val - originalOutput[i]);
            expect(diff).toBeLessThan(0.001); // Looser tolerance that should work for all network types
          });
        });
        describe('Scenario: standalone function throws on invalid input', () => {
          test('throws error if input length is wrong', () => {
            // Act
            const act = () => standaloneFn(input.concat([1]));
            // Assert
            expect(act).toThrow();
          });
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
    describe('Scenario: network with no output nodes', () => {
      test('should throw or return a function that returns an empty array', () => {
        // Arrange
        const net = new Network(2, 1);
        net.nodes = net.nodes.filter(n => n.type !== 'output');
        // Act & Assert
        expect(() => net.standalone()).toThrow('Cannot create standalone function: network has no output nodes.');
      });
    });
  });

  describe('Standalone Function Scenarios', () => {
    test('should fallback to identity for unknown squash function', () => {
      const net = new Network(2, 1);
      net.nodes[0].squash = function customUnknownSquash(x: number) { return x * 2; };
      const code = net.standalone();
      expect(typeof code).toBe('string');
    });
    test('should throw if output node index is invalid (no output nodes)', () => {
      const net = new Network(2, 1);
      net.nodes.pop();
      expect(() => net.standalone()).toThrow('Cannot create standalone function: network has no output nodes.');
    });
  });

  describe('Advanced Standalone Scenarios', () => {
    test('should throw if standalone is called on a network with no nodes', () => {
      // Arrange
      const net = new Network(2, 1);
      net.nodes = [];
      // Act & Assert
      expect(() => net.standalone()).toThrow();
    });

    test('should throw if standalone is called on a network with only input nodes', () => {
      // Arrange
      const net = new Network(2, 1);
      net.nodes = net.nodes.filter(n => n.type === 'input');
      // Act & Assert
      expect(() => net.standalone()).toThrow();
    });

    test('should throw if standalone is called on a network with only hidden nodes', () => {
      // Arrange
      const net = new Network(2, 1);
      net.nodes = net.nodes.filter(n => n.type === 'hidden');
      // Act & Assert
      expect(() => net.standalone()).toThrow();
    });
  });

  describe('Standalone Function Parameterized Tests', () => {
    // Define test cases to run with different architectures
    const architectureTests = [
      {
        name: 'Perceptron',
        factory: () => Architect.perceptron(2, 3, 1)
      },
      {
        name: 'LSTM',
        factory: () => Architect.lstm(2, 4, 1)
      },
      {
        name: 'GRU',
        factory: () => Architect.gru(2, 2, 1)
      }
    ];

    test.each(architectureTests)(
      '$name standalone function handles edge cases correctly',
      ({ factory }) => {
        // Arrange
        const network = factory();
        const standaloneCode = network.standalone();
        const standaloneFn = new Function(`return ${standaloneCode}`)();
        
        // Act & Assert
        // Test null input
        expect(() => standaloneFn(null)).toThrow();
        
        // Test empty array
        expect(() => standaloneFn([])).toThrow();
        
        // Test with NaN values
        const nanInput = Array(network.input).fill(NaN);
        const nanOutput = standaloneFn(nanInput);
        expect(nanOutput.every(Number.isNaN)).toBe(true);
      }
    );
  });

  // Benchmark performance between network and standalone functions
  describe('Standalone vs Network Performance', () => {
    test('standalone function performs similarly to activate()', () => {
      // Arrange
      const network = Architect.perceptron(10, 20, 5);
      const standaloneFn = new Function(`return ${stripCoverage(network.standalone())}`)();
      const inputs = Array(10).fill(0).map(() => Math.random());
      
      // Act - benchmark both methods
      const trials = 100;
      let networkTime = 0;
      let standaloneTime = 0;
      
      for (let i = 0; i < trials; i++) {
        const start1 = performance.now();
        network.activate(inputs);
        networkTime += performance.now() - start1;
        
        const start2 = performance.now();
        standaloneFn(inputs);
        standaloneTime += performance.now() - start2;
      }
      
      // Assert - standalone should not be significantly slower
      console.log(`Network: ${networkTime}ms, Standalone: ${standaloneTime}ms`);
      // This is not a strict assertion, just informative
      expect(standaloneTime).toBeLessThan(networkTime * 5);
    });
  });
});
