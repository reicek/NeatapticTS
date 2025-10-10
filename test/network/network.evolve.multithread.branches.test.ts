import Network from '../../src/architecture/network';
import { evolveNetwork } from '../../src/architecture/network/network.evolve';
import Multi from '../../src/multithreading/multi';
import { Workers } from '../../src/multithreading/workers/workers';
import type { TestWorker } from '../../src/multithreading/workers/node/testworker';

type TrainingSet = Parameters<typeof evolveNetwork>[0];

describe('Network.evolveNetwork multi-thread branches', () => {
  describe('Scenario: partial worker spawn failures', () => {
    it('still evolves with reduced worker pool', async () => {
      // Arrange
      const originalWorkers = Multi.workers;
      let spawnCount = 0;
      class SpawnFailureWorkers extends Workers {
        static override async getNodeTestWorker(): Promise<typeof TestWorker> {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any -- Mock class for testing
          return class MockTestWorker {
            ['worker']: unknown; // Required property to match TestWorker interface
            private readonly description: string;

            constructor(
              dataSet: number[],
              cost: { name: string },
            ) {
              this['worker'] = null; // Mock worker property
              this.description = `${cost.name}:${dataSet.length}`;
              spawnCount += 1;
              if (spawnCount === 2) throw new Error('fail second');
            }

            async evaluate(candidate: Network) {
              return candidate.nodes.length >= 0 ? 0.123 : 0.123;
            }

            terminate() {
              void this.description;
            }
          } as unknown as typeof TestWorker;
        }
      }
      Multi.workers = SpawnFailureWorkers;
      const net = new Network(1, 1, { seed: 70 });
      const trainingSet: TrainingSet = [{ input: [0.5], output: [0.6] }];
      try {
        // Act
        const result = await evolveNetwork.call(net, trainingSet, {
          iterations: 1,
          threads: 3,
        });
        // Assert
        expect(result.iterations).toBe(1);
      } finally {
        Multi.workers = originalWorkers;
      }
    });
  });

  describe('Scenario: worker evaluate rejection is caught and skipped', () => {
    it('continues draining queue after rejection', async () => {
      // Arrange
      const originalWorkers = Multi.workers;
      class RejectionWorkers extends Workers {
        static override async getNodeTestWorker(): Promise<typeof TestWorker> {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any -- Mock class for testing
          return class MockTestWorker {
            ['worker']: unknown; // Required property to match TestWorker interface
            #failOnce = true;
            private readonly description: string;

            constructor(
              dataSet: number[],
              cost: { name: string },
            ) {
              this['worker'] = null; // Mock worker property
              this.description = `${cost.name}:${dataSet.length}`;
            }

            async evaluate(candidate: Network) {
              if (this.#failOnce) {
                this.#failOnce = false;
                throw new Error('boom');
              }
              return candidate.nodes.length >= 0 ? 0.321 : 0.321;
            }

            terminate() {
              void this.description;
            }
          } as unknown as typeof TestWorker;
        }
      }
      Multi.workers = RejectionWorkers;
      const net = new Network(1, 1, { seed: 71 });
      const trainingSet: TrainingSet = [{ input: [0.2], output: [0.8] }];
      try {
        // Act
        const result = await evolveNetwork.call(net, trainingSet, {
          iterations: 1,
          threads: 2,
        });
        // Assert
        expect(result.iterations).toBe(1);
      } finally {
        Multi.workers = originalWorkers;
      }
    });
  });
});
