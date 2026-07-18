import { buildJuvenileLifecyclePolicy } from './neat.nge-juvenile.lifecycle-policy';

describe('buildJuvenileLifecyclePolicy', () => {
  it('returns a deterministic record covering every lifecycle stage', () => {
    const policy = buildJuvenileLifecyclePolicy();

    expect(Object.keys(policy.stages).sort()).toEqual([
      'adult',
      'baby',
      'embryo',
      'equilibrium',
      'juvenile',
    ]);
  });

  it('embryo and baby stages prefer CPU with GPU and workers disabled', () => {
    const { stages } = buildJuvenileLifecyclePolicy();

    expect(stages.embryo.backend).toBe('cpu');
    expect(stages.embryo.disableGPU).toBe(true);
    expect(stages.embryo.disableWorkers).toBe(true);

    expect(stages.baby.backend).toBe('cpu');
    expect(stages.baby.disableGPU).toBe(true);
    expect(stages.baby.disableWorkers).toBe(true);
  });

  it('juvenile and adult stages use auto backend with acceleration enabled', () => {
    const { stages } = buildJuvenileLifecyclePolicy();

    expect(stages.juvenile.backend).toBe('auto');
    expect(stages.juvenile.disableGPU).toBe(false);
    expect(stages.juvenile.disableWorkers).toBe(false);

    expect(stages.adult.backend).toBe('auto');
    expect(stages.adult.disableGPU).toBe(false);
    expect(stages.adult.disableWorkers).toBe(false);
  });

  it('equilibrium stage falls back to CPU with acceleration disabled', () => {
    const { stages } = buildJuvenileLifecyclePolicy();

    expect(stages.equilibrium.backend).toBe('cpu');
    expect(stages.equilibrium.disableGPU).toBe(true);
    expect(stages.equilibrium.disableWorkers).toBe(true);
  });

  it('produces independent fresh objects on each call', () => {
    const first = buildJuvenileLifecyclePolicy();
    const second = buildJuvenileLifecyclePolicy();

    expect(first).not.toBe(second);
    expect(first.stages).not.toBe(second.stages);
    expect(first.stages.baby).not.toBe(second.stages.baby);
  });
});
