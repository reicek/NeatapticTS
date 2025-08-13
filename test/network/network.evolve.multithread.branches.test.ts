/**
 * Multi-thread fitness builder branch tests:
 *  - Worker spawn failure (one worker succeeds, another fails)
 *  - Worker evaluate rejection path (caught and continues)
 */
import Network from '../../src/architecture/network';
import { evolveNetwork } from '../../src/architecture/network/network.evolve';

// Craft a mock Multi.workers with custom Worker constructor exercising branches.
const MultiMod = require('../../src/multithreading/multi').default;

describe('Network.evolveNetwork multi-thread branches', () => {
  describe('Scenario: partial worker spawn failures', () => {
    it('still evolves with reduced worker pool', async () => {
      // Arrange
      const originalWorkers = MultiMod.workers;
      let spawnCount = 0;
      MultiMod.workers = {
        async getNodeTestWorker() {
          return class TestWorker {
            private set: any;
            private meta: any;
            constructor(setSerialized: any, meta: any) {
              this.set = setSerialized;
              this.meta = meta;
              spawnCount++;
              if (spawnCount === 2) throw new Error('fail second');
            }
            evaluate(genome: any) {
              return Promise.resolve(0.123);
            }
            terminate() {
              /* no-op */
            }
          };
        },
      };
      const net = new Network(1, 1, { seed: 70 });
      const set = [{ input: [0.5], output: [0.6] }];
      // Act
      const res = await evolveNetwork.call(net, set as any, {
        iterations: 1,
        threads: 3,
      });
      MultiMod.workers = originalWorkers;
      // Assert
      expect(res.iterations).toBe(1);
    });
  });

  describe('Scenario: worker evaluate rejection is caught and skipped', () => {
    it('continues draining queue after rejection', async () => {
      // Arrange
      const originalWorkers = MultiMod.workers;
      MultiMod.workers = {
        async getNodeTestWorker() {
          return class TestWorker {
            private failOnce = true;
            constructor(_: any, __: any) {}
            evaluate(genome: any) {
              if (this.failOnce) {
                this.failOnce = false;
                return Promise.reject(new Error('boom'));
              }
              return Promise.resolve(0.321);
            }
            terminate() {}
          };
        },
      };
      const net = new Network(1, 1, { seed: 71 });
      const set = [{ input: [0.2], output: [0.8] }];
      // Act
      const res = await evolveNetwork.call(net, set as any, {
        iterations: 1,
        threads: 2,
      });
      MultiMod.workers = originalWorkers;
      // Assert
      expect(res.iterations).toBe(1);
    });
  });
});
