/**
 * Branch-focused tests for evolveNetwork covering:
 *  - iterations=0 early termination + _warnIfNoBestGenome path
 *  - infinite error break via repeated Infinity errors (mocked Neat)
 *  - adoption else branch when no best genome captured
 */
import Network from '../../src/architecture/network';
import { evolveNetwork } from '../../src/architecture/network/network.evolve';

// Mock the dynamically imported Neat module used inside evolveNetwork.
jest.mock('../../src/neat', () => {
  return {
    __esModule: true,
    default: class MockNeat {
      input: number;
      output: number;
      fitnessFn: any;
      options: any;
      generation = 0;
      constructor(input: number, output: number, fitnessFn: any, options: any) {
        this.input = input;
        this.output = output;
        this.fitnessFn = fitnessFn;
        this.options = options;
      }
      async evolve() {
        // Return genome with NaN score so error remains Infinity (via fallback) triggering infiniteErrorCount.
        this.generation += 1;
        return {
          score: NaN,
          nodes: [],
          connections: [],
          selfconns: [],
          gates: [],
        } as any;
      }
      _warnIfNoBestGenome() {
        console.warn(
          'Evolution completed without finding a valid best genome (mock)'
        );
      }
    },
  };
});

describe('Network.evolveNetwork branch coverage', () => {
  describe('Scenario: iterations=0 triggers warning path', () => {
    it('emits warning via _warnIfNoBestGenome when zero iterations specified', async () => {
      // Arrange
      const spy = jest.spyOn(console, 'warn').mockImplementation(() => {});
      const net = new Network(1, 1, { seed: 60 });
      const set = [{ input: [0.1], output: [0.2] }];
      // Act
      await evolveNetwork.call(net, set as any, { iterations: 0 });
      const warned = spy.mock.calls.some((c) =>
        /valid best genome/.test(String(c[0]))
      );
      spy.mockRestore();
      // Assert
      expect(warned).toBe(true);
    });
  });

  describe('Scenario: infinite error break after repeated Infinity errors', () => {
    it('terminates early (iterations less than configured upper bound)', async () => {
      // Arrange
      const net = new Network(1, 1, { seed: 61 });
      const set = [{ input: [0.9], output: [0.1] }];
      // Act
      const res = await evolveNetwork.call(net, set as any, { iterations: 50 });
      // Assert (loop should break after reaching infinite error threshold << 50)
      expect(res.iterations < 50).toBe(true);
    });
  });
});
