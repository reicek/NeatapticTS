import type { RacingRenderFrame } from './simulation-worker.types';

type Tier5RacingRenderFrame = RacingRenderFrame & {
  pitStatus: Int16Array;
};

interface Tier5SimulationWorkerModule {
  createTier5RacePack(): Tier5RacingRenderFrame;
  resolveReadableRadioRows(
    frame: RacingRenderFrame,
    carIndex: number,
  ): readonly number[];
}

describe('simulation worker Tier 5 six-car seam', () => {
  describe('createTier5RacePack', () => {
    it('returns agentCount = 6 for the 3v3 race pack', async () => {
      // Arrange
      const tier5SimulationWorkerModulePromise = loadTier5SimulationWorkerModule();

      // Act + Assert
      await expect(
        tier5SimulationWorkerModulePromise.then(({ createTier5RacePack }) =>
          createTier5RacePack().agentCount,
        ),
      ).resolves.toBe(6);
    });

    it('allocates tireState for 24 packed tire values', async () => {
      // Arrange
      const tier5SimulationWorkerModulePromise = loadTier5SimulationWorkerModule();

      // Act + Assert
      await expect(
        tier5SimulationWorkerModulePromise.then(({ createTier5RacePack }) =>
          createTier5RacePack().tireState.length,
        ),
      ).resolves.toBe(24);
    });

    it('keeps pitStatus packed as four scalars', async () => {
      // Arrange
      const tier5SimulationWorkerModulePromise = loadTier5SimulationWorkerModule();

      // Act + Assert
      await expect(
        tier5SimulationWorkerModulePromise.then(({ createTier5RacePack }) =>
          createTier5RacePack().pitStatus.length,
        ),
      ).resolves.toBe(4);
    });

    it('allocates radioField for 42 floats', async () => {
      // Arrange
      const tier5SimulationWorkerModulePromise = loadTier5SimulationWorkerModule();

      // Act + Assert
      await expect(
        tier5SimulationWorkerModulePromise.then(({ createTier5RacePack }) =>
          createTier5RacePack().radioField.length,
        ),
      ).resolves.toBe(42);
    });

    it('assigns carTeam as [0, 0, 0, 1, 1, 1]', async () => {
      // Arrange
      const tier5SimulationWorkerModulePromise = loadTier5SimulationWorkerModule();

      // Act + Assert
      await expect(
        tier5SimulationWorkerModulePromise.then(({ createTier5RacePack }) =>
          Array.from(createTier5RacePack().carTeam),
        ),
      ).resolves.toEqual([0, 0, 0, 1, 1, 1]);
    });
  });

  describe('resolveReadableRadioRows', () => {
    it('returns only teammate rows 1 and 2 for car 0', async () => {
      // Arrange
      const tier5SimulationWorkerModulePromise = loadTier5SimulationWorkerModule();

      // Act + Assert
      await expect(
        tier5SimulationWorkerModulePromise.then(
          ({ createTier5RacePack, resolveReadableRadioRows }) =>
            Array.from(resolveReadableRadioRows(createTier5RacePack(), 0)),
        ),
      ).resolves.toEqual([1, 2]);
    });

    it('returns only teammate rows 4 and 5 for car 3', async () => {
      // Arrange
      const tier5SimulationWorkerModulePromise = loadTier5SimulationWorkerModule();

      // Act + Assert
      await expect(
        tier5SimulationWorkerModulePromise.then(
          ({ createTier5RacePack, resolveReadableRadioRows }) =>
            Array.from(resolveReadableRadioRows(createTier5RacePack(), 3)),
        ),
      ).resolves.toEqual([4, 5]);
    });

    it('does not expose opponent rows 3, 4, and 5 to car 0', async () => {
      // Arrange
      const tier5SimulationWorkerModulePromise = loadTier5SimulationWorkerModule();

      // Act + Assert
      await expect(
        tier5SimulationWorkerModulePromise.then(
          ({ createTier5RacePack, resolveReadableRadioRows }) =>
            resolveReadableRadioRows(createTier5RacePack(), 0).some((radioRowIndex) =>
              [3, 4, 5].includes(radioRowIndex),
            ),
        ),
      ).resolves.toBe(false);
    });
  });
});

async function loadTier5SimulationWorkerModule(): Promise<Tier5SimulationWorkerModule> {
  const modulePath = './simulation-worker.tier5';
  const module = (await import(modulePath)) as Partial<Tier5SimulationWorkerModule>;

  if (typeof module.createTier5RacePack !== 'function') {
    throw new Error('Missing Tier 5 simulation-worker export: createTier5RacePack');
  }

  if (typeof module.resolveReadableRadioRows !== 'function') {
    throw new Error('Missing Tier 5 simulation-worker export: resolveReadableRadioRows');
  }

  return module as Tier5SimulationWorkerModule;
}
