import { Architect, Network, methods } from '../../src/neataptic';
import Node from '../../src/architecture/node';

// Retry failed tests
jest.retryTimes(3, { logErrorsBeforeRetry: true });

describe('Dropout & Regularization', () => {
  beforeEach(() => {
    jest.spyOn(console, 'warn').mockImplementation(() => {});
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('Dropout', () => {
    describe('Scenario: dropout during training', () => {
      describe('when dropout is applied during training', () => {
        let net: Network;
        let input: number[];
        let dropoutRate: number;
        let hiddenActivations: number[];
        let maskedCount: number;

        beforeEach(() => {
          // Arrange
          net = Architect.perceptron(2, 10, 1);
          input = [1, 1];
          dropoutRate = 0.8;
          net.dropout = dropoutRate;
          // Act
          net.activate(input, true);
          hiddenActivations = net.nodes
            .filter((n: Node) => n.type === 'hidden')
            .map((n: Node) => n.activation);
          maskedCount = hiddenActivations.filter((a) => a === 0).length;
        });

        it('at least one hidden node is masked', () => {
          // Assert
          expect(maskedCount).toBeGreaterThan(0);
        });

        it('not all hidden nodes are masked', () => {
          // Assert
          expect(maskedCount).toBeLessThan(hiddenActivations.length);
        });
      });
    });
    describe('Scenario: dropout during testing', () => {
      describe('when dropout is disabled during testing', () => {
        let net: Network;
        let input: number[];
        let hiddenActivationsTest: number[];
        let maskedCountTest: number;

        beforeEach(() => {
          // Arrange
          net = Architect.perceptron(2, 10, 1);
          input = [1, 1];
          net.dropout = 0.8;
          net.activate(input, true);
          net.dropout = 0;
          net.nodes.forEach((n: Node) => {
            if (n.type === 'hidden') n.mask = 1;
          });
          // Act
          net.activate(input, false);
          hiddenActivationsTest = net.nodes
            .filter((n: Node) => n.type === 'hidden')
            .map((n: Node) => n.activation);
          maskedCountTest = hiddenActivationsTest.filter((a) => a === 0).length;
        });

        it('no hidden node is masked', () => {
          // Assert
          expect(maskedCountTest).toBe(0);
        });

        it('all hidden nodes are active', () => {
          // Assert
          expect(hiddenActivationsTest.every((a) => a !== 0)).toBe(true);
        });
      });
    });
    describe('Scenario: dropout rate 0', () => {
      describe('when dropout rate is 0', () => {
        let net: Network;
        let input: number[];
        let hiddenActivations: number[];
        let maskedCount: number;

        beforeEach(() => {
          // Arrange
          net = Architect.perceptron(2, 10, 1);
          input = [1, 1];
          net.dropout = 0;
          // Act
          net.activate(input, true);
          hiddenActivations = net.nodes
            .filter((n: Node) => n.type === 'hidden')
            .map((n: Node) => n.activation);
          maskedCount = hiddenActivations.filter((a) => a === 0).length;
        });

        it('no hidden node is masked', () => {
          // Assert
          expect(maskedCount).toBe(0);
        });

        it('all hidden nodes are active', () => {
          // Assert
          expect(hiddenActivations.every((a) => a !== 0)).toBe(true);
        });
      });
    });
    describe('Scenario: dropout rate 1', () => {
      describe('when dropout rate is 1', () => {
        let net: Network;
        let input: number[];
        let hiddenActivations: number[];
        let maskedCount: number;

        beforeEach(() => {
          // Arrange
          net = Architect.perceptron(2, 10, 1);
          input = [1, 1];
          net.dropout = 1;
          // Act
          net.activate(input, true);
          hiddenActivations = net.nodes
            .filter((n: Node) => n.type === 'hidden')
            .map((n: Node) => n.activation);
          maskedCount = hiddenActivations.filter((a) => a === 0).length;
        });

        it('all but one hidden node are masked', () => {
          // Assert
          expect(maskedCount).toBe(hiddenActivations.length - 1);
        });

        it('at least one hidden node is active', () => {
          // Assert
          expect(hiddenActivations.some((a) => a !== 0)).toBe(true);
        });
      });
    });
  });

  describe('Dropout Mask Reset', () => {
    let net: Network;
    type DataSample = { input: number[]; output: number[] };
    const dataset: DataSample[] = [
      { input: [0, 0], output: [0] },
      { input: [0, 1], output: [1] },
      { input: [1, 0], output: [1] },
      { input: [1, 1], output: [0] },
    ];
    beforeEach(() => {
      net = Architect.perceptron(2, 4, 1);
    });

    describe('Scenario: after training', () => {
      let net: Network;
      let hiddenMasks: number[];
      const localDataset: DataSample[] = [
        { input: [0, 0], output: [0] },
        { input: [0, 1], output: [1] },
        { input: [1, 0], output: [1] },
        { input: [1, 1], output: [0] },
      ];
      beforeEach(() => {
        // Arrange
        net = Architect.perceptron(2, 10, 1);
        // Act
        net.train(localDataset, { iterations: 2, dropout: 0.5 });
        hiddenMasks = net.nodes
          .filter((n: Node) => n.type === 'hidden')
          .map((n: Node) => n.mask);
      });

      it('all hidden node masks are 1 after train()', () => {
        // Assert
        expect(hiddenMasks.every((m: number) => m === 1)).toBe(true);
      });

      it('all hidden nodes are present after train()', () => {
        // Assert
        expect(hiddenMasks.length).toBeGreaterThan(0);
      });
    });

    describe('Scenario: after testing', () => {
      describe('after calling test()', () => {
        beforeEach(() => {
          // Arrange
          net.train(dataset, { iterations: 2, dropout: 0.5 });
          // Act
          net.test(dataset);
        });

        it('all hidden node masks are 1 after test()', () => {
          // Assert
          const hiddenMasks = net.nodes
            .filter((n: Node) => n.type === 'hidden')
            .map((n: Node) => n.mask);
          expect(hiddenMasks.every((m: number) => m === 1)).toBe(true);
        });
      });

      describe('dropout affects training but not inference', () => {
        let net1: Network;
        let net2: Network;
        let out1: number[];
        let out2: number[];
        beforeEach(() => {
          // Arrange
          net1 = Architect.perceptron(2, 10, 1);
          net2 = net1.clone();
          // Act - train with and without dropout
          net1.train(dataset, { iterations: 5, dropout: 0 });
          net2.train(dataset, { iterations: 5, dropout: 0.5 });
          out1 = net1.activate([0, 1]);
          out2 = net2.activate([0, 1]);
        });

        it('outputs should differ due to different training', () => {
          // Assert
          expect(out1).not.toEqual(out2);
        });

        it('all nodes in net1 have mask=1 during inference', () => {
          // Assert
          expect(net1.nodes.every((n: Node) => n.mask === 1)).toBe(true);
        });

        it('all nodes in net2 have mask=1 during inference', () => {
          // Assert
          expect(net2.nodes.every((n: Node) => n.mask === 1)).toBe(true);
        });
      });
    });

    describe('Scenario: during training', () => {
      describe('when training with dropout', () => {
        let originalActivate: any;
        let maskedNodeSeen: boolean;
        beforeEach(() => {
          // Arrange
          originalActivate = net.activate;
          maskedNodeSeen = false;
          // Spy
          net.activate = jest.fn((...args) => {
            const isTraining = args[1] === true;
            const result = originalActivate.apply(net, args);
            if (isTraining) {
              const hiddenNodes = net.nodes.filter(
                (n: Node) => n.type === 'hidden'
              );
              if (hiddenNodes.some((n: Node) => n.mask === 0)) {
                maskedNodeSeen = true;
              }
            }
            return result;
          });
          net.train(dataset, { iterations: 5, dropout: 0.5 });
        });

        afterEach(() => {
          net.activate = originalActivate;
        });

        it('some hidden nodes are masked during training', () => {
          // Assert
          expect(maskedNodeSeen).toBe(true);
        });
      });
    });
  });

  describe('Regularization', () => {
    type DataSample = { input: number[]; output: number[] };
    const dataset: DataSample[] = [
      { input: [0, 0], output: [0] },
      { input: [0, 1], output: [1] },
      { input: [1, 0], output: [1] },
      { input: [1, 1], output: [0] },
    ];

    describe('Scenario: L2 regularization', () => {
      describe('when comparing error with and without L2 regularization', () => {
        let netNoReg: Network;
        let netReg: Network;
        let resultNoReg: any;
        let resultReg: any;
        beforeEach(() => {
          // Arrange
          netNoReg = Architect.perceptron(2, 4, 1);
          netReg = Architect.perceptron(2, 4, 1);
          for (let i = 0; i < netNoReg.connections.length; i++) {
            netReg.connections[i].weight = netNoReg.connections[i].weight;
          }
          // Act
          resultNoReg = netNoReg.train(dataset, {
            iterations: 100,
            error: 0.01,
            regularization: 0,
          });
          resultReg = netReg.train(dataset, {
            iterations: 100,
            error: 0.01,
            regularization: 10,
          });
        });

        it('resultReg.error is a number', () => {
          // Assert
          expect(typeof resultReg.error).toBe('number');
        });

        it('resultNoReg.error is a number', () => {
          // Assert
          expect(typeof resultNoReg.error).toBe('number');
        });
      });

      describe('when comparing weight magnitudes with and without L2 regularization', () => {
        let netNoReg: Network;
        let netReg: Network;
        let avgWeightNoReg: number;
        let avgWeightReg: number;
        beforeEach(() => {
          // Arrange
          netNoReg = Architect.perceptron(2, 4, 1);
          netReg = Architect.perceptron(2, 4, 1);
          for (let i = 0; i < netNoReg.connections.length; i++) {
            netNoReg.connections[i].weight = 3;
            netReg.connections[i].weight = 3;
          }
          // Act
          netNoReg.train(dataset, {
            iterations: 50,
            error: 0.01,
            regularization: 0,
          });
          netReg.train(dataset, {
            iterations: 50,
            error: 0.01,
            regularization: 10,
          });
          avgWeightNoReg =
            netNoReg.connections.reduce(
              (sum, c) => sum + Math.abs(c.weight),
              0
            ) / netNoReg.connections.length;
          avgWeightReg =
            netReg.connections.reduce((sum, c) => sum + Math.abs(c.weight), 0) /
            netReg.connections.length;
        });

        it('avgWeightReg is finite', () => {
          // Assert
          expect(Number.isFinite(avgWeightReg)).toBe(true);
        });

        it('avgWeightNoReg is finite', () => {
          // Assert
          expect(Number.isFinite(avgWeightNoReg)).toBe(true);
        });
      });
    });

    describe('Scenario: regularization 0', () => {
      describe('when regularization is 0', () => {
        let net: Network;
        let result: any;
        beforeEach(() => {
          // Arrange
          net = Architect.perceptron(2, 4, 1);
          // Act
          result = net.train(dataset, {
            iterations: 100,
            error: 0.01,
            regularization: 0,
          });
        });

        it('error is a number', () => {
          // Assert
          expect(typeof result.error).toBe('number');
        });

        it('error is not NaN', () => {
          // Assert
          expect(isNaN(result.error)).toBe(false);
        });
      });

      describe('when comparing regularization 0 and no regularization', () => {
        let netWithZero: Network;
        let netWithoutReg: Network;
        let resultWithZero: any;
        let resultWithoutReg: any;
        beforeEach(() => {
          // Arrange
          netWithZero = Architect.perceptron(2, 4, 1);
          netWithoutReg = Architect.perceptron(2, 4, 1);
          for (let i = 0; i < netWithZero.connections.length; i++) {
            const weight = Math.random() * 2 - 1;
            netWithZero.connections[i].weight = weight;
            netWithoutReg.connections[i].weight = weight;
          }
          // Act
          resultWithZero = netWithZero.train(dataset, {
            iterations: 10,
            error: 0.01,
            regularization: 0,
          });
          resultWithoutReg = netWithoutReg.train(dataset, {
            iterations: 10,
            error: 0.01,
          });
        });

        it('errors are close', () => {
          // Assert
          expect(resultWithZero.error).toBeCloseTo(resultWithoutReg.error, 2);
        });

        it('weights are updated the same way', () => {
          // Assert
          for (let i = 0; i < netWithZero.connections.length; i++) {
            expect(netWithZero.connections[i].weight).toBeCloseTo(
              netWithoutReg.connections[i].weight,
              1
            );
          }
        });
      });
    });

    describe('Scenario: L1 regularization', () => {
      describe('when comparing L1 and L2 regularization sparsity', () => {
        let netL1: Network;
        let netL2: Network;
        let zeroWeightsL1: number;
        let zeroWeightsL2: number;
        beforeEach(() => {
          // Arrange
          netL1 = Architect.perceptron(2, 10, 1);
          netL2 = Architect.perceptron(2, 10, 1);
          for (let i = 0; i < netL1.connections.length; i++) {
            const weight = Math.random() * 0.1;
            netL1.connections[i].weight = weight;
            netL2.connections[i].weight = weight;
          }
          // Act
          netL1.train(dataset, {
            iterations: 50,
            error: 0.01,
            regularization: { type: 'L1', lambda: 0.01 },
          });
          netL2.train(dataset, {
            iterations: 50,
            error: 0.01,
            regularization: { type: 'L2', lambda: 0.01 },
          });
          const zeroThreshold = 1e-4;
          zeroWeightsL1 = netL1.connections.filter(
            (c) => Math.abs(c.weight) < zeroThreshold
          ).length;
          zeroWeightsL2 = netL2.connections.filter(
            (c) => Math.abs(c.weight) < zeroThreshold
          ).length;
        });

        it('L1 regularization produces at least as many near-zero weights as L2', () => {
          // Assert
          expect(zeroWeightsL1).toBeGreaterThanOrEqual(zeroWeightsL2);
        });
      });
    });
  });

  describe('Alternative Cost Functions', () => {
    type DataSample = { input: number[]; output: number[] };
    const dataset: DataSample[] = [
      { input: [0, 0], output: [0] },
      { input: [0, 1], output: [1] },
      { input: [1, 0], output: [1] },
      { input: [1, 1], output: [0] },
    ];
    let net: Network;
    beforeEach(() => {
      net = Architect.perceptron(2, 4, 1);
    });

    describe('Scenario: MAE cost function', () => {
      describe('when using MAE cost function', () => {
        let localNet: Network;
        let result: any;
        beforeEach(() => {
          // Arrange
          localNet = Architect.perceptron(2, 4, 1);
          // Act
          result = localNet.train(dataset, {
            iterations: 100,
            error: 0.1,
            cost: methods.Cost.mae,
          });
        });

        it('train returns error less than 1', () => {
          // Assert
          expect(result.error).toBeLessThan(1);
        });
      });

      describe('when testing with MAE cost function', () => {
        let testResult: any;
        beforeEach(() => {
          // Arrange
          net.train(dataset, { iterations: 10, cost: methods.Cost.mae });
          // Act
          testResult = net.test(dataset, methods.Cost.mae);
        });

        it('test returns error less than 1', () => {
          // Assert
          expect(testResult.error).toBeLessThan(1);
        });
      });

      describe('when comparing MAE and MSE cost functions', () => {
        let netMAE: Network;
        let netMSE: Network;
        let errorMAE: number;
        let errorMSE: number;
        beforeEach(() => {
          // Arrange
          netMAE = Architect.perceptron(2, 4, 1);
          netMSE = Architect.perceptron(2, 4, 1);
          for (let i = 0; i < netMAE.connections.length; i++) {
            const weight = Math.random() * 2 - 1;
            netMAE.connections[i].weight = weight;
            netMSE.connections[i].weight = weight;
          }
          // Act
          netMAE.train(dataset, { iterations: 20, cost: methods.Cost.mae });
          netMSE.train(dataset, { iterations: 20, cost: methods.Cost.mse });
          errorMAE = netMAE.test(dataset, methods.Cost.mae).error;
          errorMSE = netMSE.test(dataset, methods.Cost.mse).error;
        });

        it('MAE and MSE cost functions produce different training results', () => {
          // Assert
          expect(Math.abs(errorMAE - errorMSE)).toBeGreaterThan(1e-3);
        });
      });
    });

    describe('Scenario: invalid cost function', () => {
      describe('when using an invalid cost function', () => {
        it('should throw', () => {
          // Arrange
          const net = Architect.perceptron(2, 4, 1);
          // Act & Assert
          expect(() =>
            net.train(dataset, {
              iterations: 100,
              error: 0.1,
              cost: 'notARealCostFn' as any,
            })
          ).toThrow();
        });
      });

      describe('when using a custom cost function', () => {
        it('custom cost function works if properly defined', () => {
          // Arrange
          const customCost = {
            calculate: (target: number[], output: number[]) => {
              return (
                target.reduce((sum, t, i) => sum + Math.abs(t - output[i]), 0) /
                target.length
              );
            },
            derivative: (target: number, output: number) => {
              return output > target ? 1 : -1;
            },
          };
          // Act & Assert
          expect(() =>
            net.train(dataset, { iterations: 5, cost: customCost })
          ).not.toThrow();
        });
      });
    });
  });

  describe('Dropout Functionality', () => {
    describe('Basic Dropout', () => {
      let net: Network;
      let hiddenMasks: number[];
      let hiddenNodes: Node[];
      beforeEach(() => {
        // Arrange
        net = new Network(2, 1);
        net.mutate(methods.mutation.ADD_NODE);
        hiddenMasks = net.nodes
          .filter((n: Node) => n.type === 'hidden')
          .map((n: Node) => n.mask);
      });

      it('all hidden nodes are present initially', () => {
        // Assert
        expect(hiddenMasks.length).toBeGreaterThan(0);
      });

      it('all masks are initially 1', () => {
        // Assert
        expect(hiddenMasks.every((m: number) => m === 1)).toBe(true);
      });

      describe('after training with dropout', () => {
        beforeEach(() => {
          // Act
          net.train([{ input: [0, 0], output: [0] }], {
            iterations: 2,
            dropout: 0.5,
          });
          hiddenNodes = net.nodes.filter((n: Node) => n.type === 'hidden');
        });

        it('hidden nodes are still present', () => {
          // Assert
          expect(hiddenNodes.length).toBeGreaterThan(0);
        });

        it('all masks are reset to 1', () => {
          // Assert
          expect(hiddenNodes.every((n: Node) => n.mask === 1)).toBe(true);
        });
      });
    });

    describe('Clone behavior with dropout', () => {
      describe('when cloning a network with dropout', () => {
        let net1: Network;
        let net2: Network;
        beforeEach(() => {
          // Arrange
          net1 = new Network(2, 1);
          net1.mutate(methods.mutation.ADD_NODE);
          net2 = net1.clone();
          // Act
          net1.train([{ input: [0, 0], output: [0] }], {
            iterations: 5,
            dropout: 0.5,
          });
        });

        it('cloned network masks are all 1', () => {
          // Assert
          expect(net2.nodes.every((n: Node) => n.mask === 1)).toBe(true);
        });
      });
    });
  });

  describe('L2 Regularization', () => {
    describe('when training on noisy data', () => {
      type DataSample = { input: number[]; output: number[] };
      const generateNoisyXORData = (
        numSamples: number,
        noiseLevel: number
      ): DataSample[] => {
        const data: DataSample[] = [];
        for (let i = 0; i < numSamples; i++) {
          const x = Math.random() > 0.5 ? 1 : 0;
          const y = Math.random() > 0.5 ? 1 : 0;
          // Target is XOR plus noise
          const target =
            (x !== y ? 1 : 0) +
            Math.random() * noiseLevel * (Math.random() > 0.5 ? 1 : -1);
          data.push({ input: [x, y], output: [target] });
        }
        return data;
      };

      let withReg: Network;
      let withoutReg: Network;
      let errorWithReg: number;
      let errorWithoutReg: number;
      let trainingData: DataSample[];
      let testData: DataSample[];

      beforeEach(() => {
        // Arrange
        trainingData = generateNoisyXORData(20, 0.2);
        testData = generateNoisyXORData(10, 0);
        // Network without regularization
        withoutReg = new Network(2, 1);
        withoutReg.mutate(methods.mutation.ADD_NODE);
        withoutReg.mutate(methods.mutation.ADD_NODE);
        withoutReg.train(trainingData, {
          iterations: 5000,
          error: 0.01,
          rate: 0.05,
        });
        // Network with regularization (lower lambda and rate for stability)
        withReg = new Network(2, 1);
        withReg.mutate(methods.mutation.ADD_NODE);
        withReg.mutate(methods.mutation.ADD_NODE);
        withReg.train(trainingData, {
          iterations: 5000,
          error: 0.01,
          rate: 0.05,
          regularization: 0.001,
          regularizationType: 'L2',
        });
        // Defensive: check for NaN weights after training
        const nanWeights = withReg.connections.filter(
          (c) => !Number.isFinite(c.weight)
        );
        if (nanWeights.length > 0) {
          throw new Error(
            `NaN/Infinity detected in weights after training with regularization: ` +
              nanWeights.map((c, i) => `conn#${i}=${c.weight}`).join(', ')
          );
        }
        // Test both networks on clean data
        errorWithReg = withReg.test(testData).error;
        errorWithoutReg = withoutReg.test(testData).error;
      });

      it('regularized network should not perform worse than unregularized (probabilistic)', () => {
        // Assert
        if (
          !Number.isFinite(errorWithReg) ||
          !Number.isFinite(errorWithoutReg)
        ) {
          throw new Error(
            `Test failed: errorWithReg=${errorWithReg}, errorWithoutReg=${errorWithoutReg}. ` +
              'One or both errors are not finite. This may indicate instability in the network or regularization logic.'
          );
        }
        // This is a probabilistic test, so we only check that regularization does not make things worse
        expect(errorWithReg).toBeLessThanOrEqual(errorWithoutReg);
      });
    });

    describe('Dropout Mask Usage During Training', () => {
      describe('when applying dropout during training and testing', () => {
        let net: Network;
        let dropoutApplied: boolean;
        let originalActivate: any;
        beforeEach(() => {
          // Arrange
          net = new Network(2, 1);
          net.mutate(methods.mutation.ADD_NODE);
          net.mutate(methods.mutation.ADD_NODE);
          dropoutApplied = false;
          // Spy
          originalActivate = net.activate;
          net.activate = function (input, training) {
            if (training && this.dropout > 0) {
              const hiddenNodes = net.nodes.filter(
                (n: Node) => n.type === 'hidden'
              );
              if (hiddenNodes.some((n: Node) => n.mask === 0)) {
                dropoutApplied = true;
              }
            }
            return originalActivate.call(this, input, training);
          };
        });

        afterEach(() => {
          net.activate = originalActivate;
        });

        it('applies dropout during training', () => {
          // Act
          net.train(
            [
              { input: [0, 0], output: [0] },
              { input: [1, 1], output: [0] },
            ],
            {
              iterations: 20,
              dropout: 0.5,
            }
          );
          // Assert
          expect(dropoutApplied).toBe(true);
        });

        it('does not apply dropout during testing', () => {
          // Arrange
          net.train(
            [
              { input: [0, 0], output: [0] },
              { input: [1, 1], output: [0] },
            ],
            {
              iterations: 20,
              dropout: 0.5,
            }
          );
          // Reset detection
          dropoutApplied = false;
          // Act
          net.test([{ input: [0, 0], output: [0] }]);
          // Assert
          expect(dropoutApplied).toBe(false);
        });
      });
    });
  });
});
