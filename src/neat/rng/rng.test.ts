import Network from '../../architecture/network';
import Neat from '../../neat';

describe('neat rng chapter', () => {
  describe('state snapshot and replay', () => {
    describe('given a seeded controller that has already consumed samples', () => {
      const scoreWithoutSideEffects = (network: Network) => {
        void network;
        return 0;
      };

      let snapshotState: number | undefined;
      let initialSequence: number[];
      let repeatedSequence: number[];
      let restoredSequence: number[];
      let replicaSequence: number[];

      beforeEach(() => {
        // Arrange
        const neat = new Neat(2, 1, scoreWithoutSideEffects, {
          popsize: 4,
          seed: 42,
        });

        neat.sampleRandom(10);
        snapshotState = neat.snapshotRNGState();
        initialSequence = neat.sampleRandom(5);

        // Act
        neat.restoreRNGState(snapshotState);
        repeatedSequence = neat.sampleRandom(5);

        const neatReplica = new Neat(2, 1, scoreWithoutSideEffects, {
          popsize: 4,
        });

        neatReplica.importRNGState(JSON.stringify(snapshotState));
        replicaSequence = neatReplica.sampleRandom(5);

        neat.restoreRNGState(snapshotState);
        restoredSequence = neat.sampleRandom(5);
      });

      describe('when the same controller restores a saved snapshot', () => {
        it('replays the same future subsequence', () => {
          // Assert
          expect(repeatedSequence).toEqual(initialSequence);
        });
      });

      describe('when a fresh controller imports the exported snapshot', () => {
        it('matches the restored original controller subsequence', () => {
          // Assert
          expect(replicaSequence).toEqual(restoredSequence);
        });
      });
    });
  });
});
