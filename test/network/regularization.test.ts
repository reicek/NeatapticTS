import { Architect, Network, methods } from '../../src/neataptic';

describe('Dropout & Regularization', () => {
  describe('Dropout', () => {
    describe('Scenario: dropout during training', () => {
      test('at least one hidden node is masked', () => {
        // Arrange
        const net = Architect.perceptron(2, 10, 1);
        const input = [1, 1];
        const dropoutRate = 0.8;
        net.dropout = dropoutRate;
        // Act
        net.activate(input, true);
        const hiddenActivations = net.nodes.filter(n => n.type === 'hidden').map(n => n.activation);
        const maskedCount = hiddenActivations.filter(a => a === 0).length;
        // Assert
        expect(maskedCount).toBeGreaterThan(0);
      });
    });
    describe('Scenario: dropout during testing', () => {
      test('no hidden node is masked', () => {
        // Arrange
        const net = Architect.perceptron(2, 10, 1);
        const input = [1, 1];
        net.dropout = 0.8;
        net.activate(input, true);
        net.dropout = 0;
        net.nodes.forEach(n => { if (n.type === 'hidden') n.mask = 1; });
        // Act
        net.activate(input, false);
        const hiddenActivationsTest = net.nodes.filter(n => n.type === 'hidden').map(n => n.activation);
        const maskedCountTest = hiddenActivationsTest.filter(a => a === 0).length;
        // Assert
        expect(maskedCountTest).toBe(0);
      });
    });
    describe('Scenario: dropout rate 0', () => {
      test('no hidden node is masked', () => {
        // Arrange
        const net = Architect.perceptron(2, 10, 1);
        const input = [1, 1];
        net.dropout = 0;
        // Act
        net.activate(input, true);
        const hiddenActivations = net.nodes.filter(n => n.type === 'hidden').map(n => n.activation);
        const maskedCount = hiddenActivations.filter(a => a === 0).length;
        // Assert
        expect(maskedCount).toBe(0);
      });
    });
    describe('Scenario: dropout rate 1', () => {
      test('all hidden nodes are masked', () => {
        // Arrange
        const net = Architect.perceptron(2, 10, 1);
        const input = [1, 1];
        net.dropout = 1;
        // Act
        net.activate(input, true);
        const hiddenActivations = net.nodes.filter(n => n.type === 'hidden').map(n => n.activation);
        const maskedCount = hiddenActivations.filter(a => a === 0).length;
        // Assert
        expect(maskedCount).toBe(hiddenActivations.length);
      });
    });
  });

  describe('Regularization', () => {
    describe('Scenario: L2 regularization', () => {
      test('error with regularization is greater than or equal to no regularization', () => {
        // Arrange
        const dataset = [
          { input: [0, 0], output: [0] },
          { input: [0, 1], output: [1] },
          { input: [1, 0], output: [1] },
          { input: [1, 1], output: [0] },
        ];
        const netNoReg = Architect.perceptron(2, 4, 1);
        const resultNoReg = netNoReg.train(dataset, { iterations: 100, error: 0.01, regularization: 0 });
        const netReg = Architect.perceptron(2, 4, 1);
        const resultReg = netReg.train(dataset, { iterations: 100, error: 0.01, regularization: 10 });
        // Assert
        expect(resultReg.error).toBeGreaterThanOrEqual(resultNoReg.error);
      });
    });
    describe('Scenario: regularization 0', () => {
      test('error is a number', () => {
        // Arrange
        const dataset = [
          { input: [0, 0], output: [0] },
          { input: [0, 1], output: [1] },
          { input: [1, 0], output: [1] },
          { input: [1, 1], output: [0] },
        ];
        const net = Architect.perceptron(2, 4, 1);
        // Act
        const result = net.train(dataset, { iterations: 100, error: 0.01, regularization: 0 });
        // Assert
        expect(typeof result.error).toBe('number');
      });
    });
  });

  describe('Alternative Cost Functions', () => {
    describe('Scenario: MAE cost function', () => {
      test('train returns error less than 1', () => {
        // Arrange
        const dataset = [
          { input: [0, 0], output: [0] },
          { input: [0, 1], output: [1] },
          { input: [1, 0], output: [1] },
          { input: [1, 1], output: [0] },
        ];
        const net = Architect.perceptron(2, 4, 1);
        // Act
        const result = net.train(dataset, { iterations: 100, error: 0.1, cost: methods.Cost.mae });
        // Assert
        expect(result.error).toBeLessThan(1);
      });
      test('test returns error less than 1', () => {
        // Arrange
        const dataset = [
          { input: [0, 0], output: [0] },
          { input: [0, 1], output: [1] },
          { input: [1, 0], output: [1] },
          { input: [1, 1], output: [0] },
        ];
        const net = Architect.perceptron(2, 4, 1);
        net.train(dataset, { iterations: 100, error: 0.1, cost: methods.Cost.mae });
        // Act
        const testResult = net.test(dataset, methods.Cost.mae);
        // Assert
        expect(testResult.error).toBeLessThan(1);
      });
    });
    describe('Scenario: invalid cost function', () => {
      test('should throw or return NaN', () => {
        // Arrange
        const dataset = [
          { input: [0, 0], output: [0] },
          { input: [0, 1], output: [1] },
          { input: [1, 0], output: [1] },
          { input: [1, 1], output: [0] },
        ];
        const net = Architect.perceptron(2, 4, 1);
        // Act & Assert
        expect(() => net.train(dataset, { iterations: 100, error: 0.1, cost: 'notARealCostFn' as any })).toThrow();
      });
    });
  });
});
