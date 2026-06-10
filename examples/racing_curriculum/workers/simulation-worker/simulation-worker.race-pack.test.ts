/**
 * Red-phase contracts for the missing deterministic race-pack factory in
 * `simulation-worker.race-pack.service.ts`.
 *
 * Contracts verified here:
 * - Same seed + same opponent snapshot → identical initial conditions
 * - Packed `race-step` frames use the shared `'racing-packed-v1'` schema
 * - Transfer-list ownership includes the expected typed-array buffers
 * - Detached buffers cannot be reused after transfer
 * - Car starting positions are not all zero (cars spread on the grid)
 *
 * All tests stay red until Step 04 implements the service boundary.
 * Single-expect rule is enforced throughout.
 *
 * TODO: NGE_TODO — When NGE EpisodicSlot and GatingRouter primitives become
 * available (upstream Phase G/E), the opponent snapshot payload format should
 * be extended to include episodic context and hard task-switch state.
 */
import type { RacingRenderFrame } from './simulation-worker.types';

// ---------------------------------------------------------------------------
// Locally-defined interface
// ---------------------------------------------------------------------------

type OpponentSnapshot = {
  /** Stable identifier frozen at snapshot capture time. */
  readonly snapshotId: string;
  /** Generation index at which this snapshot was captured. */
  readonly generation: number;
  /** Serialised network payloads for opponent controllers. */
  readonly networkPayloads: readonly unknown[];
};

interface RacePackService {
  /**
   * Constructs an initial race frame deterministically from a seed and a
   * frozen opponent snapshot.  Identical inputs must return identical frames.
   */
  createDeterministicRacePack(
    seed: number,
    opponentSnapshot: OpponentSnapshot,
  ): RacingRenderFrame;

  /**
   * Collects every ArrayBuffer backing a typed-array field in the frame into a
   * transfer list suitable for zero-copy postMessage transfer.
   * Mirrors the existing `resolveRacingRenderFrameTransferList` contract but is
   * owned by this service boundary.
   */
  resolveRaceStepTransferList(frame: RacingRenderFrame): ArrayBuffer[];
}

// ---------------------------------------------------------------------------
// Module loader
// ---------------------------------------------------------------------------

async function loadRacePackService(): Promise<RacePackService> {
  const modulePath = './simulation-worker.race-pack.service';
  return (await import(modulePath)) as RacePackService;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeMinimalOpponentSnapshot(): OpponentSnapshot {
  return {
    snapshotId: 'snap-test-0',
    generation: 1,
    networkPayloads: [],
  };
}

// ---------------------------------------------------------------------------
// Red tests
// ---------------------------------------------------------------------------

describe('simulation worker deterministic race-pack service', () => {
  describe('createDeterministicRacePack determinism', () => {
    it('returns identical car-X positions for identical seed and opponent snapshot', async () => {
      // Arrange
      const service = await loadRacePackService();
      const snapshot = makeMinimalOpponentSnapshot();

      // Act
      const packA = service.createDeterministicRacePack(42, snapshot);
      const packB = service.createDeterministicRacePack(42, snapshot);

      // Assert — same seed + same snapshot must produce identical initial car positions
      expect(Array.from(packA.carX)).toEqual(Array.from(packB.carX));
    });

    it('places at least one car at a non-zero X position to confirm grid spread', async () => {
      // Arrange
      const service = await loadRacePackService();
      const snapshot = makeMinimalOpponentSnapshot();

      // Act
      const pack = service.createDeterministicRacePack(99, snapshot);

      // Assert — cars must not all start at the origin
      expect(pack.carX.some((position) => position !== 0)).toBe(true);
    });

    it('emits the shared racing-packed-v1 schema sentinel for packed race-step frames', async () => {
      // Arrange
      const service = await loadRacePackService();
      const snapshot = makeMinimalOpponentSnapshot();

      // Act
      const pack = service.createDeterministicRacePack(11, snapshot);

      // Assert
      expect(pack.schemaVersion).toBe('racing-packed-v1');
    });
  });

  describe('resolveRaceStepTransferList buffer ownership', () => {
    it('includes the expected typed-array buffers exactly once for a packed race-step frame', async () => {
      // Arrange
      const service = await loadRacePackService();
      const pack = service.createDeterministicRacePack(
        7,
        makeMinimalOpponentSnapshot(),
      );

      // Act
      const transferList = service.resolveRaceStepTransferList(pack);
      const uniqueBufferCount = new Set(transferList).size;

      // Assert — ownership contract requires every typed-array buffer exactly once
      expect({
        containsCarXBuffer: transferList.includes(
          pack.carX.buffer as ArrayBuffer,
        ),
        entryCount: transferList.length,
        uniqueBufferCount,
      }).toEqual({
        containsCarXBuffer: true,
        entryCount: 10,
        uniqueBufferCount: 10,
      });
    });

    it('detaches the carX buffer after a simulated transfer using the resolved list', async () => {
      // Arrange
      const service = await loadRacePackService();
      const pack = service.createDeterministicRacePack(
        7,
        makeMinimalOpponentSnapshot(),
      );
      const transferList = service.resolveRaceStepTransferList(pack);

      // Act
      structuredClone(pack, { transfer: transferList });

      // Assert
      expect(pack.carX.byteLength).toBe(0);
    });
  });
});
