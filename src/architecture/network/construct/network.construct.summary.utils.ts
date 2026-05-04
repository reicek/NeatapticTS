import type {
  ConstructGraphConnectionSummary,
  ConstructGraphNodeSummary,
  ConstructResult,
} from './network.construct.utils.types';

const MAX_CONNECTION_PREVIEW_COUNT = 12;
const MAX_ACTIVATION_ORDER_PREVIEW_COUNT = 16;

/**
 * Build a compact human-readable summary for one construct-from-parts result.
 *
 * The summary is layered on top of the detached `ConstructResult.graph`
 * snapshot so tooling can log or display one stable explanation of the built
 * graph without reading mutable `Network` internals.
 *
 * @param constructResult Construct result returned by `Network.construct(...)`.
 * @returns Multi-line summary string suitable for logs, diagnostics panels, or snapshots.
 *
 * @example
 * ```ts
 * const construction = Network.construct([sensor, hidden, readout]);
 * const summary = formatConstructSummary(construction);
 * ```
 */
export function formatConstructSummary(
  constructResult: ConstructResult,
): string {
  const { diagnostics, graph } = constructResult;
  const inputNodes = graph.nodes.filter((node) => node.inputOrder !== null);
  const outputNodes = graph.nodes.filter((node) => node.outputOrder !== null);
  const roleCounts = resolveRoleCounts(graph.nodes);

  // Step 1: Build the stable header lines from the detached graph snapshot.
  const summaryLines = [
    'Construct summary',
    `Mode: ${graph.requestedMode} (${graph.topologyIntent})`,
    `Graph: ${graph.nodes.length} nodes, ${graph.connections.length} connections, cycles detected: ${diagnostics.detectedCycles ? 'yes' : 'no'}`,
    `Roles: ${roleCounts.input} input, ${roleCounts.hidden} hidden, ${roleCounts.output} output`,
    `Inputs: ${formatRoleNodeList(inputNodes, 'input')}`,
    `Outputs: ${formatRoleNodeList(outputNodes, 'output')}`,
    `Activation order: ${formatActivationOrder(graph.activationOrder)}`,
    `Connections: ${formatConnectionPreview(graph.connections)}`,
  ];

  // Step 2: Join into one log-friendly multi-line block.
  return summaryLines.join('\n');
}

function resolveRoleCounts(
  nodes: readonly ConstructGraphNodeSummary[],
): Record<'input' | 'hidden' | 'output', number> {
  return nodes.reduce<Record<'input' | 'hidden' | 'output', number>>(
    (counts, node) => {
      if (
        node.role === 'input' ||
        node.role === 'hidden' ||
        node.role === 'output'
      ) {
        counts[node.role] += 1;
      }

      return counts;
    },
    { input: 0, hidden: 0, output: 0 },
  );
}

function formatRoleNodeList(
  nodes: readonly ConstructGraphNodeSummary[],
  roleLabel: 'input' | 'output',
): string {
  if (!nodes.length) {
    return `none resolved for ${roleLabel} role`;
  }

  const orderedNodes = nodes.toSorted((leftNode, rightNode) => {
    const leftOrder =
      roleLabel === 'input' ? leftNode.inputOrder : leftNode.outputOrder;
    const rightOrder =
      roleLabel === 'input' ? rightNode.inputOrder : rightNode.outputOrder;

    return (
      (leftOrder ?? Number.POSITIVE_INFINITY) -
      (rightOrder ?? Number.POSITIVE_INFINITY)
    );
  });

  return orderedNodes
    .map((node) => {
      const roleOrder =
        roleLabel === 'input' ? node.inputOrder : node.outputOrder;
      return `[${roleOrder}] ${formatNodeIdentity(node)}`;
    })
    .join(', ');
}

function formatActivationOrder(activationOrder: readonly number[]): string {
  if (!activationOrder.length) {
    return 'none';
  }

  const activationPreview = activationOrder.slice(
    0,
    MAX_ACTIVATION_ORDER_PREVIEW_COUNT,
  );
  const activationSuffix =
    activationOrder.length > MAX_ACTIVATION_ORDER_PREVIEW_COUNT
      ? ` -> ... (+${activationOrder.length - MAX_ACTIVATION_ORDER_PREVIEW_COUNT} more)`
      : '';

  return `${activationPreview.join(' -> ')}${activationSuffix}`;
}

function formatConnectionPreview(
  connections: readonly ConstructGraphConnectionSummary[],
): string {
  if (!connections.length) {
    return 'none';
  }

  const connectionPreview = connections.slice(0, MAX_CONNECTION_PREVIEW_COUNT);
  const connectionSuffix =
    connections.length > MAX_CONNECTION_PREVIEW_COUNT
      ? `, ... (+${connections.length - MAX_CONNECTION_PREVIEW_COUNT} more)`
      : '';

  return `${connectionPreview
    .map((connection, connectionIndex) => {
      const fromReference = formatConnectionEndpoint(
        connection.fromGeneId,
        connection.fromIndex,
      );
      const toReference = formatConnectionEndpoint(
        connection.toGeneId,
        connection.toIndex,
      );
      const selfEdgeSuffix = connection.isSelfConnection ? ' [self]' : '';
      const gaterSuffix =
        connection.gaterGeneId === null
          ? ''
          : ` [gater: geneId:${connection.gaterGeneId}]`;

      return `[${connectionIndex}] ${fromReference} -> ${toReference}${selfEdgeSuffix}${gaterSuffix}`;
    })
    .join(', ')}${connectionSuffix}`;
}

function formatNodeIdentity(node: ConstructGraphNodeSummary): string {
  const nodeName =
    node.label === null ? `geneId:${node.geneId}` : `"${node.label}"`;
  return `${nodeName} (geneId: ${node.geneId})`;
}

function formatConnectionEndpoint(
  nodeGeneId: number,
  nodeIndex: number,
): string {
  return `geneId:${nodeGeneId}[${nodeIndex}]`;
}
