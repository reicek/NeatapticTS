import { assert } from 'chai';
import { Architect, Network, methods } from '../src/neataptic';
import mocha from 'mocha';

/**
 * Checks if a mutation method modifies the network's output.
 * @param {any} method - The mutation method to test.
 */
function checkMutation(method: any): void {
  const network = Architect.perceptron(2, 4, 4, 4, 2);
  network.mutate(methods.mutation.ADD_GATE);
  network.mutate(methods.mutation.ADD_BACK_CONN);
  network.mutate(methods.mutation.ADD_SELF_CONN);

  const originalOutput: number[][] = [];
  for (let i = 0; i <= 10; i++) {
    for (let j = 0; j <= 10; j++) {
      originalOutput.push(network.activate([i / 10, j / 10]));
    }
  }

  network.mutate(method);

  const mutatedOutput: number[][] = [];
  for (let i = 0; i <= 10; i++) {
    for (let j = 0; j <= 10; j++) {
      mutatedOutput.push(network.activate([i / 10, j / 10]));
    }
  }

  assert.notDeepEqual(
    originalOutput,
    mutatedOutput,
    'Output of original network should be different from the mutated network!'
  );
}

/**
 * Trains a network on a given dataset and checks if the error is below a threshold.
 * @param {Array<{ input: number[]; output: number[] }>} set - The training dataset.
 * @param {number} iterations - The number of training iterations.
 * @param {number} error - The acceptable error threshold.
 */
function learnSet(
  set: Array<{ input: number[]; output: number[] }>,
  iterations: number,
  error: number
): void {
  const network = Architect.perceptron(
    set[0].input.length,
    5,
    set[0].output.length
  );

  const options = {
    iterations: iterations,
    error: error,
    shuffle: true,
    rate: 0.3,
    momentum: 0.9,
  };

  const results = network.train(set, options);

  assert.isBelow(results.error, error);
}

/**
 * Tests if two networks (or a network and its standalone function) produce the same output.
 * @param {Network} original - The original network.
 * @param {Network | Function} copied - The copied network or standalone function.
 */
function testEquality(original: Network, copied: Network | Function): void {
  for (let j = 0; j < 50; j++) {
    const input: number[] = [];
    for (let a = 0; a < original.input; a++) {
      input.push(Math.random());
    }

    const ORout = original.activate(input);
    const COout =
      copied instanceof Network ? copied.activate(input) : copied(input);

    for (let a = 0; a < original.output; a++) {
      ORout[a] = parseFloat(ORout[a].toFixed(9));
      COout[a] = parseFloat(COout[a].toFixed(9));
    }
    assert.deepEqual(
      ORout,
      COout,
      copied instanceof Network
        ? 'Original and JSON copied networks are not the same!'
        : 'Original and standalone networks are not the same!'
    );
  }
}

/*******************************************************************************************
                          Test the performance of networks
*******************************************************************************************/

describe('Networks', function () {
  describe('Mutation', function () {
    it('ADD_NODE', function () {
      checkMutation(methods.mutation.ADD_NODE);
    });
    it('ADD_CONNECTION', function () {
      checkMutation(methods.mutation.ADD_CONN);
    });
    it('MOD_BIAS', function () {
      checkMutation(methods.mutation.MOD_BIAS);
    });
    it('MOD_WEIGHT', function () {
      checkMutation(methods.mutation.MOD_WEIGHT);
    });
    it('SUB_CONN', function () {
      checkMutation(methods.mutation.SUB_CONN);
    });
    it('SUB_NODE', function () {
      checkMutation(methods.mutation.SUB_NODE);
    });
    it('MOD_ACTIVATION', function () {
      checkMutation(methods.mutation.MOD_ACTIVATION);
    });
    it('ADD_GATE', function () {
      checkMutation(methods.mutation.ADD_GATE);
    });
    it('SUB_GATE', function () {
      checkMutation(methods.mutation.SUB_GATE);
    });
    it('ADD_SELF_CONN', function () {
      checkMutation(methods.mutation.ADD_SELF_CONN);
    });
    it('SUB_SELF_CONN', function () {
      checkMutation(methods.mutation.SUB_SELF_CONN);
    });
    it('ADD_BACK_CONN', function () {
      checkMutation(methods.mutation.ADD_BACK_CONN);
    });
    it('SUB_BACK_CONN', function () {
      checkMutation(methods.mutation.SUB_BACK_CONN);
    });
    it('SWAP_NODES', function () {
      checkMutation(methods.mutation.SWAP_NODES);
    });
  });

  describe('Structure', function () {
    it('Feed-forward', function () {
      this.timeout(30000);
      const network1 = new Network(2, 2);
      const network2 = new Network(2, 2);

      // Mutate the networks
      for (let i = 0; i < 100; i++) {
        network1.mutate(methods.mutation.ADD_NODE);
        network2.mutate(methods.mutation.ADD_NODE);
      }
      for (let i = 0; i < 400; i++) {
        network1.mutate(methods.mutation.ADD_CONN);
        network2.mutate(methods.mutation.ADD_NODE);
      }

      // Crossover
      const network = Network.crossOver(network1, network2) as Network;

      // Check if the network is feed-forward correctly
      for (const connection of network.connections) {
        const from = network.nodes.indexOf(connection.from);
        const to = network.nodes.indexOf(connection.to);

        assert.isBelow(from, to, 'Network is not feeding forward correctly');
      }
    });

    it('from/toJSON equivalency', function () {
      this.timeout(10000);
      let original: Network, copy: Network;

      original = Architect.perceptron(
        Math.floor(Math.random() * 5 + 1),
        Math.floor(Math.random() * 5 + 1),
        Math.floor(Math.random() * 5 + 1)
      );
      copy = Network.fromJSON(original.toJSON() as any);
      testEquality(original, copy);
    });

    it('standalone equivalency', function () {
      this.timeout(10000);
      let original: Network;
      original = Architect.perceptron(
        Math.floor(Math.random() * 5 + 1),
        Math.floor(Math.random() * 5 + 1),
        Math.floor(Math.random() * 5 + 1)
      );
      const standaloneCode = original.standalone() as string;
      const activate = new Function(`return ${standaloneCode}`)();
      testEquality(original, activate);
    });

    it('from/toJSON equivalency (extended)', function () {
      this.timeout(10000);
      const architectures = [
        function () {
          return Architect.perceptron(3, 5, 2);
        },
        function () {
          return Architect.lstm(3, 5, 2);
        },
        function () {
          return Architect.gru(3, 5, 2);
        },
        function () {
          return Architect.random(3, 10, 2);
        },
        function () {
          return Architect.narx(3, 5, 2, 3, 3);
        },
        function () {
          return Architect.hopfield(5);
        },
      ];

      architectures.forEach(function (createNetwork) {
        const original = createNetwork();
        const copy = Network.fromJSON(original.toJSON());
        testEquality(original, copy);
      });
    });
  });

  describe('Standalone Function Generation', function () {
    it('should generate valid JavaScript', function () {
      const network = new Network(2, 2);
      const standaloneCode = network.standalone();
      assert.doesNotThrow(() => {
        new Function(`return ${standaloneCode}`)();
      });
    });
  });

  describe('Learning capability', function () {
    it('AND gate', function () {
      learnSet(
        [
          { input: [0, 0], output: [0] },
          { input: [0, 1], output: [0] },
          { input: [1, 0], output: [0] },
          { input: [1, 1], output: [1] },
        ],
        1000,
        0.002
      );
    });

    it('XOR gate', function () {
      learnSet(
        [
          { input: [0, 0], output: [0] },
          { input: [0, 1], output: [1] },
          { input: [1, 0], output: [1] },
          { input: [1, 1], output: [0] },
        ],
        3000,
        0.002
      );
    });

    it('SIN function', function () {
      this.timeout(30000);
      const set = Array.from({ length: 100 }, () => {
        const inputValue = Math.random() * Math.PI * 2;
        return {
          input: [inputValue / (Math.PI * 2)],
          output: [(Math.sin(inputValue) + 1) / 2],
        };
      });

      learnSet(set, 1000, 0.05);
    });

    it('Bigger than', function () {
      this.timeout(30000);
      const set = Array.from({ length: 100 }, () => {
        const x = Math.random();
        const y = Math.random();
        return { input: [x, y], output: [x > y ? 1 : 0] };
      });

      learnSet(set, 500, 0.05);
    });

    it('SIN + COS', function () {
      this.timeout(30000);
      const set = Array.from({ length: 100 }, () => {
        const inputValue = Math.random() * Math.PI * 2;
        return {
          input: [inputValue / (Math.PI * 2)],
          output: [
            (Math.sin(inputValue) + 1) / 2,
            (Math.cos(inputValue) + 1) / 2,
          ],
        };
      });

      learnSet(set, 1000, 0.05);
    });

    it('SHIFT', function () {
      const set = Array.from({ length: 1000 }, () => {
        const x = Math.random();
        const y = Math.random();
        const z = Math.random();
        return { input: [x, y, z], output: [z, x, y] };
      });

      learnSet(set, 500, 0.03);
    });
  });
});
