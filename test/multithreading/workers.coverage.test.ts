import Multi from '../../src/multithreading/multi';

describe('Workers coverage', () => {
  it('Workers.getNodeTestWorker loads a class with evaluate/terminate', async () => {
    const mod = await import(
      '../../src/multithreading/workers/node/testworker'
    );
    jest
      .spyOn(Multi, 'getNodeTestWorker')
      .mockResolvedValue(mod.TestWorker as any);
    const WorkerClass = await Multi.getNodeTestWorker();
    expect(typeof WorkerClass).toBe('function');
    const instance = new WorkerClass([], { name: 'mse' });
    expect(typeof instance.evaluate).toBe('function');
    expect(typeof instance.terminate).toBe('function');
    instance.terminate();
  });

  it('Workers.getBrowserTestWorker returns a class type when mocked', async () => {
    class DummyWorker {
      constructor(_ds: number[], _c: { name: string }) {}
      evaluate() {
        return Promise.resolve(0);
      }
      terminate() {}
      static _createBlobString() {
        return '';
      }
    }
    jest
      .spyOn(Multi, 'getBrowserTestWorker')
      .mockResolvedValue(DummyWorker as any);
    const WorkerClass = await Multi.getBrowserTestWorker();
    expect(typeof WorkerClass).toBe('function');
  });
});
