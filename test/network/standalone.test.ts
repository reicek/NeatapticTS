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
          // Assert
          const outputsAreEqual = standaloneOutput.every(
            (val, i) =>
              typeof originalOutput[i] === 'number' &&
              Math.abs(val - originalOutput[i]) < 1e-9
          );
          expect(outputsAreEqual).toBe(true);
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
});
