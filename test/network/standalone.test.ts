describe('stripCoverage utility', () => {
  it('removes istanbul ignore comments', () => {
    // Arrange
    const code = '/* istanbul ignore next */\nfunction foo() {}';
    // Act
    const result = stripCoverage(code);
    // Assert
    expect(result).not.toMatch(/istanbul/);
  });
  it('removes coverage counters', () => {
    // Arrange
    const code = 'cov_123().s[0]++; cov_123().f[1]++; cov_123()';
    // Act
    const result = stripCoverage(code);
    // Assert
    expect(result).not.toMatch(/cov_123/);
  });
  it('removes sourceMappingURL comments', () => {
    // Arrange
    const code = '//# sourceMappingURL=foo.js';
    // Act
    const result = stripCoverage(code);
    // Assert
    expect(result).not.toMatch(/sourceMappingURL/);
  });
  it('removes extra semicolons and commas', () => {
    // Arrange
    const code = ';;,,';
    // Act
    const result = stripCoverage(code);
    // Assert
    expect(result.trim()).toBe('');
  });
});
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
  code = code.replace(/^[\t ]*[,;]+\s*$/gm, '');
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
        describe('Output shape and value', () => {
          it('produces the same output length as the original', () => {
            // Assert
            expect(standaloneOutput.length).toEqual(originalOutput.length);
          });
          (standaloneOutput || []).forEach((val, i) => {
            it(`output[${i}] is numerically close to original`, () => {
              // Arrange
              // Act
              const originalVal = original.activate(input)[i];
              const standaloneVal = standaloneFn(input)[i];
              // Assert
              expect(standaloneVal).toBeCloseTo(originalVal, 3);
            });
          });
        });
        describe('Scenario: standalone function throws on invalid input', () => {
          it('throws error if input length is wrong', () => {
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
      it('should throw or return a function that returns an empty array', () => {
        // Arrange
        const net = new Network(2, 1);
        net.nodes = net.nodes.filter((n) => n.type !== 'output');
        // Act & Assert
        expect(() => net.standalone()).toThrow(
          'Cannot create standalone function: network has no output nodes.'
        );
      });
    });
  });

  describe('Standalone Function Scenarios', () => {
    describe('Scenario: fallback to identity for unknown squash function', () => {
      let net: Network;
      let code: string;
      beforeAll(() => {
        // Arrange
        net = new Network(2, 1);
        net.nodes[0].squash = function customUnknownSquash(x) {
          return x * 2;
        };
        code = net.standalone();
      });
      it('returns a string for standalone code', () => {
        // Assert
        expect(typeof code).toBe('string');
      });
    });
    describe('Scenario: throw if output node index is invalid (no output nodes)', () => {
      let net: Network;
      beforeAll(() => {
        // Arrange
        net = new Network(2, 1);
        net.nodes.pop();
      });
      it('throws with correct error message', () => {
        // Act
        const act = () => net.standalone();
        // Assert
        expect(act).toThrow(
          'Cannot create standalone function: network has no output nodes.'
        );
      });
    });
  });

  describe('Advanced Standalone Scenarios', () => {
    describe('Scenario: standalone is called on a network with no nodes', () => {
      let net: Network;
      beforeAll(() => {
        // Arrange
        net = new Network(2, 1);
        net.nodes = [];
      });
      it('throws an error', () => {
        // Act
        const act = () => net.standalone();
        // Assert
        expect(act).toThrow();
      });
    });
    describe('Scenario: standalone is called on a network with only input nodes', () => {
      let net: Network;
      beforeAll(() => {
        // Arrange
        net = new Network(2, 1);
        net.nodes = net.nodes.filter((n) => n.type === 'input');
      });
      it('throws an error', () => {
        // Act
        const act = () => net.standalone();
        // Assert
        expect(act).toThrow();
      });
    });
    describe('Scenario: standalone is called on a network with only hidden nodes', () => {
      let net: Network;
      beforeAll(() => {
        // Arrange
        net = new Network(2, 1);
        net.nodes = net.nodes.filter((n) => n.type === 'hidden');
      });
      it('throws an error', () => {
        // Act
        const act = () => net.standalone();
        // Assert
        expect(act).toThrow();
      });
    });
  });

  describe('Standalone Function Parameterized Tests', () => {
    // Define test cases to run with different architectures
    const architectureTests = [
      {
        name: 'Perceptron',
        factory: () => Architect.perceptron(2, 3, 1),
      },
      {
        name: 'LSTM',
        factory: () => Architect.lstm(2, 4, 1),
      },
      {
        name: 'GRU',
        factory: () => Architect.gru(2, 2, 1),
      },
    ];

    architectureTests.forEach(({ name, factory }) => {
      describe(`Scenario: ${name} standalone function input validation`, () => {
        let network: Network;
        let standaloneFn: (input: number[] | null) => number[];
        beforeAll(() => {
          // Arrange
          network = factory();
          const standaloneCode = network.standalone();
          standaloneFn = new Function(`return ${standaloneCode}`)();
        });
        it('throws on null input', () => {
          // Act
          const act = () => standaloneFn(null);
          // Assert
          expect(act).toThrow();
        });
        it('throws on empty array input', () => {
          // Act
          const act = () => standaloneFn([]);
          // Assert
          expect(act).toThrow();
        });
        it('returns all NaN for NaN input', () => {
          // Arrange
          const nanInput = Array(network.input).fill(NaN);
          // Act
          const nanOutput = standaloneFn(nanInput);
          // Assert
          nanOutput.forEach((val: number) =>
            expect(Number.isNaN(val)).toBe(true)
          );
        });
      });
    });
  });

  // Benchmark performance between network and standalone functions
  describe('Standalone vs Network Performance', () => {
    it('standalone function is not significantly slower than activate()', () => {
      // Arrange
      const network = Architect.perceptron(10, 20, 5);
      const standaloneFn = new Function(
        `return ${stripCoverage(network.standalone())}`
      )();
      const inputs = Array(10)
        .fill(0)
        .map(() => Math.random());
      // Act
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
      // Assert
      // Suppress log, only assert
      expect(standaloneTime).toBeLessThan(networkTime * 5);
    });
  });
});
