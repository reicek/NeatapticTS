import { cp } from 'child_process';
import path from 'path';

/** TestWorker Class */
export class TestWorker {
  constructor(dataSet, cost) {
    this.worker = cp.fork(path.join(__dirname, '/worker'));
    this.worker.send({ set: dataSet, cost: cost.name });
  }

  evaluate(network) {
    return new Promise((resolve) => {
      const serialized = network.serialize();

      const data = {
        activations: serialized[0],
        states: serialized[1],
        conns: serialized[2],
      };

      const _that = this.worker;
      this.worker.on('message', function callback(e) {
        _that.removeListener('message', callback);
        resolve(e);
      });

      this.worker.send(data);
    });
  }

  terminate() {
    this.worker.kill();
  }
}
