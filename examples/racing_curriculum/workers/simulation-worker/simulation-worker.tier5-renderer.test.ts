import {
  assertRacingSchemaVersion,
  resolveRacingRenderFrameTransferList,
} from './simulation-worker.snapshot.utils';
import type { RacingRenderFrame } from './simulation-worker.types';

type Tier5RacingRenderFrame = RacingRenderFrame & {
  pitStatus: Int16Array;
};

interface Tier5SimulationWorkerModule {
  createTier5RacePack(): Tier5RacingRenderFrame;
}

describe('simulation worker Tier 5 renderer snapshot seam', () => {
  describe('RacingRenderFrame compatibility', () => {
    it('packs tireState for 24 values when agentCount = 6', async () => {
      // Arrange
      const tier5SimulationWorkerModulePromise =
        loadTier5SimulationWorkerModule();

      // Act + Assert
      await expect(
        tier5SimulationWorkerModulePromise.then(
          ({ createTier5RacePack }) => createTier5RacePack().tireState.length,
        ),
      ).resolves.toBe(24);
    });

    it('keeps pitStatus as Int16Array(6) for Tier 5 frames', async () => {
      // Arrange
      const tier5SimulationWorkerModulePromise =
        loadTier5SimulationWorkerModule();

      // Act + Assert
      await expect(
        tier5SimulationWorkerModulePromise.then(({ createTier5RacePack }) => {
          const racePack = createTier5RacePack();
          return (
            racePack.pitStatus instanceof Int16Array &&
            racePack.pitStatus.length === 6
          );
        }),
      ).resolves.toBe(true);
    });

    it('creates a snapshot-safe Tier 5 race pack without schema assertion errors', async () => {
      // Arrange
      const tier5SimulationWorkerModulePromise =
        loadTier5SimulationWorkerModule();

      // Act + Assert
      await expect(
        tier5SimulationWorkerModulePromise.then(({ createTier5RacePack }) => {
          const racePack = createTier5RacePack();
          assertRacingSchemaVersion(racePack);
          return resolveRacingRenderFrameTransferList(racePack).length;
        }),
      ).resolves.toBe(11);
    });

    it('includes explicit focused-car metadata for six-car renderer parity', async () => {
      // Arrange
      const tier5SimulationWorkerModulePromise =
        loadTier5SimulationWorkerModule();

      // Act + Assert
      await expect(
        tier5SimulationWorkerModulePromise.then(({ createTier5RacePack }) => {
          const racePack = createTier5RacePack() as unknown as Record<
            string,
            unknown
          >;
          return 'focusCarIndex' in racePack;
        }),
      ).resolves.toBe(true);
    });
  });
});

async function loadTier5SimulationWorkerModule(): Promise<Tier5SimulationWorkerModule> {
  const modulePath = './simulation-worker.tier5';
  const module = (await import(
    modulePath
  )) as Partial<Tier5SimulationWorkerModule>;

  if (typeof module.createTier5RacePack !== 'function') {
    throw new Error(
      'Missing Tier 5 simulation-worker export: createTier5RacePack',
    );
  }

  return module as Tier5SimulationWorkerModule;
}
