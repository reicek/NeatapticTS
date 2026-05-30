import type { RacingRenderFrame } from './simulation-worker.types';

type Tier4RacingRenderFrame = RacingRenderFrame & {
  pitStatus: Uint8Array | Uint16Array;
};

interface Tier4SimulationWorkerModule {
  createTier4RacePack(): Tier4RacingRenderFrame;
}

describe('simulation worker Tier 4 pit-status seam', () => {
  describe('createTier4RacePack', () => {
    it('returns the canonical 2v2 Tier 4 layout', async () => {
      // Arrange + Act + Assert
      await expect(
        loadTier4SimulationWorkerModule().then(({ createTier4RacePack }) => {
          const racePack = createTier4RacePack();
          return {
            carTeam: Array.from(racePack.carTeam),
            radioFieldLength: racePack.radioField.length,
            tireStateLength: racePack.tireState.length,
          };
        }),
      ).resolves.toEqual({
        carTeam: [0, 0, 1, 1],
        radioFieldLength: 28,
        tireStateLength: 16,
      });
    });

    it('adds a pitStatus field to RacingRenderFrame when Tier 4 is active', async () => {
      // Arrange + Act + Assert
      await expect(
        loadTier4SimulationWorkerModule().then(({ createTier4RacePack }) =>
          'pitStatus' in createTier4RacePack(),
        ),
      ).resolves.toBe(true);
    });

    it('packs pitStatus as four scalars for both teams', async () => {
      // Arrange + Act + Assert
      await expect(
        loadTier4SimulationWorkerModule().then(({ createTier4RacePack }) =>
          createTier4RacePack().pitStatus.length,
        ),
      ).resolves.toBe(4);
    });
  });
});

async function loadTier4SimulationWorkerModule(): Promise<Tier4SimulationWorkerModule> {
  const modulePath = './simulation-worker.tier4';
  return (await import(modulePath)) as Tier4SimulationWorkerModule;
}
