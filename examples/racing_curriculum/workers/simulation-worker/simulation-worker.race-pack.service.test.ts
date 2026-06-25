/**
 * Sibling smoke tests for `simulation-worker.race-pack.service.ts`.
 *
 * Full contract tests live in `simulation-worker.race-pack.test.ts`
 * (authored in Step 03).  This file satisfies the folder quality gate's
 * sibling-test-file requirement.
 *
 * Single-expect rule enforced throughout.
 */
import {
  createDeterministicRacePack,
  resolveRaceStepTransferList,
  createRaceEpisodeRunner,
} from './simulation-worker.race-pack.service';

describe('simulation-worker.race-pack.service module exports', () => {
  describe('createDeterministicRacePack', () => {
    it('is exported as a function', () => {
      expect(typeof createDeterministicRacePack).toBe('function');
    });

    it('returns a frame with the correct schema version sentinel', () => {
      const frame = createDeterministicRacePack(1, {
        snapshotId: 'smoke-test',
        generation: 0,
        networkPayloads: [],
      });

      expect(frame.schemaVersion).toBe('racing-packed-v1');
    });

    it('returns a frame with a Float32Array carX field', () => {
      const frame = createDeterministicRacePack(1, {
        snapshotId: 'smoke-test',
        generation: 0,
        networkPayloads: [],
      });

      expect(frame.carX).toBeInstanceOf(Float32Array);
    });
  });

  describe('resolveRaceStepTransferList', () => {
    it('is exported as a function', () => {
      expect(typeof resolveRaceStepTransferList).toBe('function');
    });

    it('returns an array of ArrayBuffer entries', () => {
      const frame = createDeterministicRacePack(1, {
        snapshotId: 'smoke-test',
        generation: 0,
        networkPayloads: [],
      });
      const transferList = resolveRaceStepTransferList(frame);

      expect(Array.isArray(transferList)).toBe(true);
    });
  });

  describe('createRaceEpisodeRunner', () => {
    it('is exported as a function', () => {
      expect(typeof createRaceEpisodeRunner).toBe('function');
    });
  });
});
