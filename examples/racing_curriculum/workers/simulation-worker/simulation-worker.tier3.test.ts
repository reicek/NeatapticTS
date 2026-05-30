import type { RacingRenderFrame } from './simulation-worker.types';

interface Tier3SimulationWorkerModule {
  createTier3RacePack(): RacingRenderFrame;
  resolveReadableRadioRows(
    frame: RacingRenderFrame,
    carIndex: number,
  ): readonly number[];
}

/**
 * Red-phase contracts for the missing Tier 3 four-car simulation-worker seam.
 *
 * The worker does not yet expose a dedicated 2v2 race-pack boundary, so these
 * tests intentionally stay red until that seam exists.
 */
describe('Tier 3 two-team simulation worker seam', () => {
  describe('race pack initialization', () => {
    it('initializes with agentCount = 4 for 2v2 Tier 3', async () => {
      await expect(
        loadTier3SimulationWorkerModule().then(({ createTier3RacePack }) =>
          createTier3RacePack().agentCount,
        ),
      ).resolves.toBe(4);
    });

    it('assigns carTeam [0, 0, 1, 1] for two cars per team', async () => {
      await expect(
        loadTier3SimulationWorkerModule().then(({ createTier3RacePack }) =>
          Array.from(createTier3RacePack().carTeam),
        ),
      ).resolves.toEqual([0, 0, 1, 1]);
    });
  });

  describe('radio field layout', () => {
    it('allocates radioField of length 28 for 4 agents × 7 channels', async () => {
      await expect(
        loadTier3SimulationWorkerModule().then(({ createTier3RacePack }) =>
          createTier3RacePack().radioField.length,
        ),
      ).resolves.toBe(28);
    });

    it('keeps Team A radio rows at indices 0 and 1 (channels 0–13)', async () => {
      await expect(
        loadTier3SimulationWorkerModule().then(
          ({ createTier3RacePack, resolveReadableRadioRows }) => {
            const racePack = createTier3RacePack();
            return {
              readableRows: Array.from(resolveReadableRadioRows(racePack, 0)),
              radioWindow: [0, 13],
            };
          },
        ),
      ).resolves.toEqual({
        readableRows: [0, 1],
        radioWindow: [0, 13],
      });
    });

    it('keeps Team B radio rows at indices 2 and 3 (channels 14–27)', async () => {
      await expect(
        loadTier3SimulationWorkerModule().then(
          ({ createTier3RacePack, resolveReadableRadioRows }) => {
            const racePack = createTier3RacePack();
            return {
              readableRows: Array.from(resolveReadableRadioRows(racePack, 2)),
              radioWindow: [14, 27],
            };
          },
        ),
      ).resolves.toEqual({
        readableRows: [2, 3],
        radioWindow: [14, 27],
      });
    });
  });
});

async function loadTier3SimulationWorkerModule(): Promise<Tier3SimulationWorkerModule> {
  const modulePath = './simulation-worker.tier3';
  return (await import(modulePath)) as Tier3SimulationWorkerModule;
}
