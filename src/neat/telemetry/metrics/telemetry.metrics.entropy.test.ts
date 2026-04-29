import { computeDegreeCounts } from './telemetry.metrics.entropy';

jest.retryTimes(2, { logErrorsBeforeRetry: true });

function createGraph(
  nodes: Array<{ geneId: number }>,
  connections: Array<{
    from: { geneId: number };
    to: { geneId: number };
    enabled: boolean;
  }>,
) {
  return { nodes, connections };
}

describe('telemetry metrics entropy chapter', () => {
  describe('computeDegreeCounts()', () => {
    describe('given a disabled connection', () => {
      it('skips the disabled connection and leaves both node degrees at zero', () => {
        // Arrange
        const graph = createGraph(
          [{ geneId: 1 }, { geneId: 2 }],
          [{ from: { geneId: 1 }, to: { geneId: 2 }, enabled: false }],
        );

        // Act
        const result = computeDegreeCounts(graph);

        // Assert: disabled connection contributes no degree
        expect(result).toEqual({ 1: 0, 2: 0 });
      });
    });

    describe('given a connection whose source geneId is absent from the node list', () => {
      it('leaves the source degree unchanged while still counting the target', () => {
        // Arrange: node list has only geneId=2; connection from geneId=99 (unknown)
        const graph = createGraph(
          [{ geneId: 2 }],
          [{ from: { geneId: 99 }, to: { geneId: 2 }, enabled: true }],
        );

        // Act
        const result = computeDegreeCounts(graph);

        // Assert: source 99 is ignored; target 2 is incremented
        expect(result).toEqual({ 2: 1 });
      });
    });

    describe('given a connection whose target geneId is absent from the node list', () => {
      it('counts the source degree while leaving the unknown target out', () => {
        // Arrange: node list has only geneId=1; connection to geneId=99 (unknown)
        const graph = createGraph(
          [{ geneId: 1 }],
          [{ from: { geneId: 1 }, to: { geneId: 99 }, enabled: true }],
        );

        // Act
        const result = computeDegreeCounts(graph);

        // Assert: source 1 is incremented; target 99 is ignored
        expect(result).toEqual({ 1: 1 });
      });
    });
  });
});
