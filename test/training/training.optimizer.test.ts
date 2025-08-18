import Network from '../../src/architecture/network';

// Simple dataset: y = 0 for x = 1
const set = [{ input: [1], output: [0] }];

const optimizers: Array<any> = ['sgd', 'rmsprop', 'adagrad', 'adam', 'adamw'];

describe('training.optimizer', () => {
  optimizers.forEach((opt) => {
    describe(opt, () => {
      let weightChanged = false;
  let hasFirstMoment = false;
  let hasSecondMoment = false;
  let hasAccumulator = false;
      beforeAll(() => {
        const net = new Network(1, 1);
        const conn: any = net.connections[0];
        const w0 = conn.weight;
        net.train(set, {
          iterations: 1,
          rate: 0.2,
          batchSize: 1,
          optimizer: opt,
        });
        weightChanged = conn.weight !== w0;
  hasFirstMoment = typeof conn.firstMoment !== 'undefined';
  hasSecondMoment = typeof conn.secondMoment !== 'undefined';
  hasAccumulator = typeof conn.gradientAccumulator !== 'undefined';
      });
      it('updates weight', () => {
        expect(weightChanged).toBe(true);
      });
      if (opt === 'adam' || opt === 'adamw') {
        it('tracks first moment', () => {
          expect(hasFirstMoment).toBe(true);
        });
        it('tracks second moment', () => {
          expect(hasSecondMoment).toBe(true);
        });
      }
      if (opt === 'rmsprop' || opt === 'adagrad') {
        it('tracks accumulator', () => {
          expect(hasAccumulator).toBe(true);
        });
      }
    });
  });
});
