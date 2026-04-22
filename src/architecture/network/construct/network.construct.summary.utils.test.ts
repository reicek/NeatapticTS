import Network from '../network';
import { formatConstructSummary } from './network.construct.summary.utils';
import type {
  ConstructGraphConnectionSummary,
  ConstructGraphNodeSummary,
  ConstructResult,
} from './network.construct.utils.types';

type ConstructSummaryFixture = {
  activationOrder?: number[];
  connections?: ConstructGraphConnectionSummary[];
  detectedCycles?: boolean;
  nodes: ConstructGraphNodeSummary[];
  requestedMode?: ConstructResult['graph']['requestedMode'];
  topologyIntent?: ConstructResult['graph']['topologyIntent'];
};

function createNodeSummary(
  overrides: Partial<ConstructGraphNodeSummary>,
): ConstructGraphNodeSummary {
  return {
    index: overrides.index ?? 0,
    geneId: overrides.geneId ?? 0,
    label: overrides.label ?? null,
    role: overrides.role ?? 'hidden',
    inputOrder: overrides.inputOrder ?? null,
    outputOrder: overrides.outputOrder ?? null,
  };
}

function createConnectionSummary(
  overrides: Partial<ConstructGraphConnectionSummary>,
): ConstructGraphConnectionSummary {
  return {
    innovation: overrides.innovation ?? 0,
    fromIndex: overrides.fromIndex ?? 0,
    toIndex: overrides.toIndex ?? 0,
    fromGeneId: overrides.fromGeneId ?? 0,
    toGeneId: overrides.toGeneId ?? 0,
    gaterGeneId: overrides.gaterGeneId ?? null,
    isSelfConnection: overrides.isSelfConnection ?? false,
    enabled: overrides.enabled ?? true,
    weight: overrides.weight ?? 1,
  };
}

function createConstructSummaryFixture(
  input: ConstructSummaryFixture,
): ConstructResult {
  const activationOrder = input.activationOrder ?? [];
  const connections = input.connections ?? [];

  return {
    network: new Network(1, 1, {}),
    diagnostics: {
      nodeCount: input.nodes.length,
      edgeCount: connections.length,
      detectedCycles: input.detectedCycles ?? false,
      activationOrder,
    },
    graph: {
      requestedMode: input.requestedMode ?? 'acyclic',
      topologyIntent: input.topologyIntent ?? 'feed-forward',
      inputNodeIds: input.nodes
        .filter((node) => node.inputOrder !== null)
        .map((node) => node.geneId),
      outputNodeIds: input.nodes
        .filter((node) => node.outputOrder !== null)
        .map((node) => node.geneId),
      activationOrder,
      nodes: input.nodes,
      connections,
    },
  };
}

describe('network construct summary chapter', () => {
  describe('formatConstructSummary()', () => {
    describe('given no public inputs, outputs, activation order, or connections are resolved', () => {
      it('returns the empty fallback summary lines', () => {
        // Arrange
        const construction = createConstructSummaryFixture({
          nodes: [
            createNodeSummary({
              index: 0,
              geneId: 7,
              role: 'bridge',
            }),
          ],
        });
        const expectedSummary = [
          'Construct summary',
          'Mode: acyclic (feed-forward)',
          'Graph: 1 nodes, 0 connections, cycles detected: no',
          'Roles: 0 input, 0 hidden, 0 output',
          'Inputs: none resolved for input role',
          'Outputs: none resolved for output role',
          'Activation order: none',
          'Connections: none',
        ].join('\n');

        // Act
        const actualSummary = formatConstructSummary(construction);

        // Assert
        expect(actualSummary).toBe(expectedSummary);
      });
    });

    describe('given output nodes include a malformed unresolved order', () => {
      it('sorts the ordered outputs first and leaves the unresolved output at the end', () => {
        // Arrange
        const construction = createConstructSummaryFixture({
          nodes: [
            {
              index: 0,
              geneId: 20,
              label: 'fallbackOutput',
              role: 'output',
              inputOrder: null,
              outputOrder: undefined as never,
            },
            createNodeSummary({
              index: 1,
              geneId: 21,
              label: 'lateOutput',
              role: 'output',
              outputOrder: 2,
            }),
            createNodeSummary({
              index: 2,
              geneId: 22,
              label: 'firstOutput',
              role: 'output',
              outputOrder: 0,
            }),
          ],
        });

        // Act
        const actualSummary = formatConstructSummary(construction);

        // Assert
        expect(actualSummary).toContain(
          'Outputs: [0] "firstOutput" (geneId: 22), [2] "lateOutput" (geneId: 21), [undefined] "fallbackOutput" (geneId: 20)',
        );
      });
    });

    describe('given input nodes include a malformed unresolved order after a valid input', () => {
      it('sorts the ordered input first and leaves the unresolved input at the end', () => {
        // Arrange
        const construction = createConstructSummaryFixture({
          nodes: [
            createNodeSummary({
              index: 0,
              geneId: 23,
              label: 'firstInput',
              role: 'input',
              inputOrder: 0,
            }),
            {
              index: 1,
              geneId: 24,
              label: 'fallbackInput',
              role: 'input',
              inputOrder: undefined as never,
              outputOrder: null,
            },
          ],
        });

        // Act
        const actualSummary = formatConstructSummary(construction);

        // Assert
        expect(actualSummary).toContain(
          'Inputs: [0] "firstInput" (geneId: 23), [undefined] "fallbackInput" (geneId: 24)',
        );
      });
    });

    describe('given the activation order is longer than the preview cap', () => {
      it('appends the activation truncation suffix', () => {
        // Arrange
        const construction = createConstructSummaryFixture({
          activationOrder: Array.from({ length: 18 }, (_unused, index) => index),
          nodes: [createNodeSummary({ index: 0, geneId: 31, role: 'input' })],
        });

        // Act
        const actualSummary = formatConstructSummary(construction);

        // Assert
        expect(actualSummary).toContain(
          'Activation order: 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9 -> 10 -> 11 -> 12 -> 13 -> 14 -> 15 -> ... (+2 more)',
        );
      });
    });

    describe('given the connection list exceeds the preview cap and includes self-gated edges', () => {
      it('appends the connection truncation suffix and formats self and gater markers', () => {
        // Arrange
        const construction = createConstructSummaryFixture({
          nodes: [createNodeSummary({ index: 0, geneId: 40, role: 'hidden' })],
          connections: Array.from(
            { length: 13 },
            (_unused, connectionIndex) =>
              createConnectionSummary({
                innovation: connectionIndex,
                fromIndex: 0,
                toIndex: 0,
                fromGeneId: 40,
                toGeneId: 40,
                gaterGeneId: connectionIndex === 0 ? 99 : null,
                isSelfConnection: connectionIndex === 0,
              }),
          ),
        });

        // Act
        const actualSummary = formatConstructSummary(construction);

        // Assert
        expect(actualSummary).toContain(
          'Connections: [0] geneId:40[0] -> geneId:40[0] [self] [gater: geneId:99]'
            + ', [1] geneId:40[0] -> geneId:40[0]'
            + ', [2] geneId:40[0] -> geneId:40[0]'
            + ', [3] geneId:40[0] -> geneId:40[0]'
            + ', [4] geneId:40[0] -> geneId:40[0]'
            + ', [5] geneId:40[0] -> geneId:40[0]'
            + ', [6] geneId:40[0] -> geneId:40[0]'
            + ', [7] geneId:40[0] -> geneId:40[0]'
            + ', [8] geneId:40[0] -> geneId:40[0]'
            + ', [9] geneId:40[0] -> geneId:40[0]'
            + ', [10] geneId:40[0] -> geneId:40[0]'
            + ', [11] geneId:40[0] -> geneId:40[0], ... (+1 more)',
        );
      });
    });
  });
});