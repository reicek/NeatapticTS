/**
 * Integration smoke test for the public acceleration layer API.
 *
 * This suite verifies that the barrel export surface is wired correctly and that
 * the {@link AccelerationManager} lifecycle can be driven end-to-end without
 * touching real GPU or worker resources.
 */

import {
  AccelerationManager,
  autoEnableAcceleration,
  shouldAutoEnableGpu,
  shouldAutoEnableWorker,
} from './index';

describe('acceleration integration', () => {
  it('exports the auto-enable orchestrator from the barrel', () => {
    expect(autoEnableAcceleration).toBeInstanceOf(Function);
  });

  it('exports the lifecycle manager from the barrel', () => {
    expect(AccelerationManager).toBeInstanceOf(Function);
  });

  it('exports the GPU eligibility helper from the barrel', () => {
    expect(shouldAutoEnableGpu).toBeInstanceOf(Function);
  });

  it('exports the worker eligibility helper from the barrel', () => {
    expect(shouldAutoEnableWorker).toBeInstanceOf(Function);
  });

  it('runs the manager lifecycle end-to-end on the CPU fallback path', async () => {
    const manager = new AccelerationManager({
      config: { disableGPU: true, disableWorkers: true, backend: 'cpu' },
    });

    const status = await manager.init(2048);

    expect(status).toMatchObject({
      mode: 'cpu',
      gpu: expect.objectContaining({ available: false }),
      worker: expect.objectContaining({ available: false }),
      cpu: expect.objectContaining({ available: true }),
    });
  });

  it('returns the same status object from getStatus() after init()', async () => {
    const manager = new AccelerationManager({
      config: { disableGPU: true, disableWorkers: true, backend: 'cpu' },
    });

    const status = await manager.init(2048);

    expect(manager.getStatus()).toBe(status);
  });

  it('tears down the manager and resets the status to CPU fallback', async () => {
    const manager = new AccelerationManager({
      config: { disableGPU: true, disableWorkers: true, backend: 'cpu' },
    });

    await manager.init(2048);
    await manager.teardown();

    expect(manager.getStatus().mode).toBe('cpu');
  });

  it('resolves cpu through the standalone auto-enable orchestrator', async () => {
    const status = await autoEnableAcceleration({
      nodeCount: 2048,
      config: { disableGPU: true, disableWorkers: true },
    });

    expect(status.mode).toBe('cpu');
  });
});
