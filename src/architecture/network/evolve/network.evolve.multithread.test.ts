import Multi from '../../../multithreading/multi';
import type { TestWorker } from '../../../multithreading/workers/node/testworker';
import { Workers } from '../../../multithreading/workers/workers';
import Network from '../network';

type TrainingSet = Parameters<Network['evolve']>[0];

describe('network evolve multithread branches', () => {
  describe('Network.evolve()', () => {
    describe('given worker discovery fails', () => {
      describe('when threads greater than one are requested', () => {
        it('falls back to single-threaded evolution', async () => {
          // Arrange
          const originalWorkers = Multi.workers;

          class DisabledWorkers extends Workers {
            static getNodeTestWorker(): Promise<never> {
              return Promise.reject(new Error('worker disabled for test'));
            }

            static getBrowserTestWorker(): Promise<never> {
              return Promise.reject(new Error('worker disabled for test'));
            }
          }

          Multi.workers = DisabledWorkers;
          const network = new Network(1, 1, { seed: 475 });
          const trainingSet: TrainingSet = [{ input: [0.3], output: [0.7] }];

          try {
            // Act
            const evolutionSummary = await network.evolve(trainingSet, {
              iterations: 1,
              threads: 4,
            });

            // Assert
            expect(evolutionSummary.iterations).toBe(1);
          } finally {
            Multi.workers = originalWorkers;
          }
        });
      });
    });

    describe('given only part of the worker pool spawns successfully', () => {
      describe('when evolve() is called', () => {
        it('still completes with the reduced worker pool', async () => {
          // Arrange
          const originalWorkers = Multi.workers;
          let spawnCount = 0;

          class SpawnFailureWorkers extends Workers {
            static getNodeTestWorker(): Promise<typeof TestWorker> {
              const MockTestWorker = class MockTestWorker {
                ['worker']: unknown;
                private readonly description: string;

                constructor(dataSet: number[], cost: { name: string }) {
                  this['worker'] = null;
                  this.description = `${cost.name}:${dataSet.length}`;
                  spawnCount += 1;
                  if (spawnCount === 2) {
                    throw new Error('fail second');
                  }
                }

                async evaluate(candidate: Network) {
                  return candidate.nodes.length >= 0 ? 0.123 : 0.123;
                }

                terminate() {
                  void this.description;
                }
              } as unknown as typeof TestWorker;

              return Promise.resolve(MockTestWorker);
            }
          }

          Multi.workers = SpawnFailureWorkers;
          const network = new Network(1, 1, { seed: 476 });
          const trainingSet: TrainingSet = [{ input: [0.5], output: [0.6] }];

          try {
            // Act
            const evolutionSummary = await network.evolve(trainingSet, {
              iterations: 1,
              threads: 3,
            });

            // Assert
            expect(evolutionSummary.iterations).toBe(1);
          } finally {
            Multi.workers = originalWorkers;
          }
        });
      });
    });

    describe('given one worker evaluation rejects', () => {
      describe('when evolve() is called', () => {
        it('continues draining the queue after the rejection', async () => {
          // Arrange
          const originalWorkers = Multi.workers;

          class RejectionWorkers extends Workers {
            static getNodeTestWorker(): Promise<typeof TestWorker> {
              const MockTestWorker = class MockTestWorker {
                ['worker']: unknown;
                #failOnce = true;
                private readonly description: string;

                constructor(dataSet: number[], cost: { name: string }) {
                  this['worker'] = null;
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

              return Promise.resolve(MockTestWorker);
            }
          }

          Multi.workers = RejectionWorkers;
          const network = new Network(1, 1, { seed: 477 });
          const trainingSet: TrainingSet = [{ input: [0.2], output: [0.8] }];

          try {
            // Act
            const evolutionSummary = await network.evolve(trainingSet, {
              iterations: 1,
              threads: 2,
            });

            // Assert
            expect(evolutionSummary.iterations).toBe(1);
          } finally {
            Multi.workers = originalWorkers;
          }
        });
      });
    });
  });
});
