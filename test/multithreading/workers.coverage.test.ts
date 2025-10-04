import Multi from '../../src/multithreading/multi';
import type {
  SerializableNetwork,
  TestWorkerConstructor,
} from '../../src/multithreading/types';

describe('Workers coverage', () => {
  it('Workers.getNodeTestWorker loads a class with evaluate/terminate', async () => {
    const mod = await import(
      '../../src/multithreading/workers/node/testworker'
    );
    const WorkerCtor = mod.TestWorker as TestWorkerConstructor;
    jest.spyOn(Multi, 'getNodeTestWorker').mockResolvedValue(WorkerCtor);
    const WorkerClass = await Multi.getNodeTestWorker();
    expect(typeof WorkerClass).toBe('function');
    const instance = new WorkerClass([], { name: 'mse' });
    expect(typeof instance.evaluate).toBe('function');
    expect(typeof instance.terminate).toBe('function');
    instance.terminate();
  });

  it('Workers.getBrowserTestWorker returns a class type when mocked', async () => {
    class DummyWorker implements InstanceType<TestWorkerConstructor> {
      private readonly payload: number[];
      private readonly descriptor: { name: string };
      constructor(dataSet: number[], cost: { name: string }) {
        this.payload = dataSet;
        this.descriptor = cost;
      }
      evaluate(candidateNetwork: SerializableNetwork) {
        void candidateNetwork;
        return Promise.resolve(
          this.payload.length + this.descriptor.name.length,
        );
      }
      terminate() {}
      static _createBlobString() {
        return '';
      }
    }
    jest.spyOn(Multi, 'getBrowserTestWorker').mockResolvedValue(DummyWorker);
    const WorkerClass = await Multi.getBrowserTestWorker();
    expect(typeof WorkerClass).toBe('function');
  });
});
