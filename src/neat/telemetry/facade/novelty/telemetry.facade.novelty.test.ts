import {
  getNoveltyArchiveSize,
  resetNoveltyArchive,
  type TelemetryFacadeNoveltyHost,
} from './telemetry.facade.novelty';

describe('neat telemetry facade novelty chapter', () => {
  describe('getNoveltyArchiveSize', () => {
    describe('given the host already retains novelty descriptors', () => {
      it('reports the current archive length', () => {
        // Arrange
        const host: TelemetryFacadeNoveltyHost = {
          _noveltyArchive: [[1], [2], [3]],
        };

        // Act
        const archiveSize = getNoveltyArchiveSize(host);

        // Assert
        expect(archiveSize).toBe(3);
      });
    });
  });

  describe('resetNoveltyArchive', () => {
    describe('given the host already retains novelty descriptors', () => {
      it('clears the novelty archive for a fresh observation window', () => {
        // Arrange
        const host: TelemetryFacadeNoveltyHost = {
          _noveltyArchive: [[1], [2], [3]],
        };

        // Act
        resetNoveltyArchive(host);
        const archiveSize = getNoveltyArchiveSize(host);

        // Assert
        expect(archiveSize).toBe(0);
      });
    });
  });
});
