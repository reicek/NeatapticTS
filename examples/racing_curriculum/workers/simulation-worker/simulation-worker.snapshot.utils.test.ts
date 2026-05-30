import type { RacingRenderFrame } from './simulation-worker.types';
import { resolveRacingRenderFrameTransferList } from './simulation-worker.snapshot.utils';

describe('simulation worker snapshot utility seam', () => {
  describe('resolveRacingRenderFrameTransferList', () => {
    it('collects one transfer entry per typed field in a solo frame', () => {
      const racingRenderFrame: RacingRenderFrame = {
        schemaVersion: 'racing-packed-v1',
        tick: 1,
        seed: 42,
        trackId: 7,
        agentCount: 1,
        featureFlags: 0,
        carX: new Float32Array([1]),
        carY: new Float32Array([2]),
        carHeading: new Float32Array([3]),
        carActive: new Uint8Array([1]),
        carTeam: new Uint8Array([0]),
        carMode: new Uint8Array([0]),
        tireState: new Float32Array([0, 0, 0, 0]),
        radioField: new Float32Array(0),
        lap: new Uint16Array([0]),
        place: new Uint8Array([1]),
        raceTimeMs: 0,
        done: false,
      };

      expect(resolveRacingRenderFrameTransferList(racingRenderFrame)).toHaveLength(
        10,
      );
    });
  });
});