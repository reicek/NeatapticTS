import Network from '../../src/architecture/network';
import { exportToONNX, importFromONNX } from '../../src/architecture/onnx';

/**
 * Round-trip numerical equivalence tests for ONNX export/import.
 * Ensures (export -> import) preserves forward activation outputs within a tight tolerance.
 * Each test follows AAA (Arrange, Act, Assert) and has a single expectation.
 */

describe('ONNX Export/Import Round-Trip Numerical Equivalence', () => {
  /**
   * Generates a random MLP, assigns deterministic pseudo-random weights/biases
   * (seeded via simple LCG) to ensure reproducibility, then validates that
   * outputs match after export/import given a fixed random input vector.
   */
  const buildRandomizedMLP = (
    input: number,
    hidden: number[],
    output: number,
    seed = 42
  ) => {
    const net = Network.createMLP(input, hidden, output);
    // Simple linear congruential generator for reproducible pseudo-random numbers
    let state = seed >>> 0;
    const rand = () => {
      state = (state * 1664525 + 1013904223) >>> 0;
      return (state & 0xffffffff) / 0xffffffff;
    };
    // Assign weights & biases deterministically
    net.connections.forEach((c: any, idx: number) => {
      c.weight = (rand() * 2 - 1) * 0.5 + idx * 1e-6; // slight variation
    });
    net.nodes.forEach((n: any, idx: number) => {
      if (n.type !== 'input') n.bias = (rand() * 2 - 1) * 0.1 + idx * 1e-6;
    });
    return net;
  };

  describe('3-4-2 MLP single sample', () => {
    it('preserves forward outputs (MSE < 1e-12)', () => {
      // Arrange
      const net = buildRandomizedMLP(3, [4], 2, 123);
      const input = [0.25, -0.5, 0.9];
      const originalOut = net.activate(input, false) as number[];
      const onnx = exportToONNX(net); // default options
      const imported = importFromONNX(onnx);
      // Act
      const importedOut = imported.activate(input, false) as number[];
      const mse =
        originalOut.reduce((acc, v, i) => {
          const d = v - importedOut[i];
          return acc + d * d;
        }, 0) / originalOut.length;
      // Assert
      expect(mse).toBeLessThan(1e-12);
    });
  });

  describe('5-6-5-3 MLP single sample', () => {
    it('preserves forward outputs (MSE < 1e-12)', () => {
      // Arrange
      const net = buildRandomizedMLP(5, [6, 5], 3, 999);
      const input = [0.1, -0.2, 0.3, -0.4, 0.5];
      const originalOut = net.activate(input, false) as number[];
      const onnx = exportToONNX(net, {
        includeMetadata: true,
        batchDimension: true,
      });
      const imported = importFromONNX(onnx);
      // Act
      const importedOut = imported.activate(input, false) as number[];
      const mse =
        originalOut.reduce((acc, v, i) => {
          const d = v - importedOut[i];
          return acc + d * d;
        }, 0) / originalOut.length;
      // Assert
      expect(mse).toBeLessThan(1e-12);
    });
  });
});
