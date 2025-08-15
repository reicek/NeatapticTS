import { Workers } from '../../src/multithreading/workers/workers';

describe('Workers dynamic imports', () => {
  it('loads node test worker class', async () => {
    const Cls = await Workers.getNodeTestWorker();
    expect(typeof Cls).toBe('function');
  });
});
