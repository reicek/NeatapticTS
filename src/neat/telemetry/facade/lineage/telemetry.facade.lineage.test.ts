import {
  getLineageSnapshot,
  type TelemetryFacadeLineageHost,
} from './telemetry.facade.lineage';

function createTelemetryLineageHost(): TelemetryFacadeLineageHost {
  return {
    population: [
      { _id: 11, _parents: [1, 2] },
      { _id: 12, _parents: [] },
      { _id: 13, _parents: [11, 12] },
    ],
  };
}

describe('neat telemetry facade lineage chapter', () => {
  describe('getLineageSnapshot', () => {
    describe('given a current population with recorded parents beyond the requested clip limit', () => {
      it('returns the clipped id-and-parents projection in population order', () => {
        // Arrange
        const telemetryLineageHost = createTelemetryLineageHost();

        // Act
        const lineageSnapshot = getLineageSnapshot(telemetryLineageHost, 2);

        // Assert
        expect(lineageSnapshot).toEqual([
          { id: 11, parents: [1, 2] },
          { id: 12, parents: [] },
        ]);
      });
    });

    describe('given the clip limit is omitted', () => {
      describe('when the helper is called directly', () => {
        it('uses the default lineage snapshot window', () => {
          // Arrange
          const telemetryLineageHost = createTelemetryLineageHost();

          // Act
          const lineageSnapshot = getLineageSnapshot(telemetryLineageHost);

          // Assert
          expect(lineageSnapshot).toEqual([
            { id: 11, parents: [1, 2] },
            { id: 12, parents: [] },
            { id: 13, parents: [11, 12] },
          ]);
        });
      });
    });
  });
});
