import Network from '../../../../src/architecture/network';
import { FLAPPY_LIGHT_NEON_RAMP } from '../../constants/constants';
import type {
  VisualNetworkConnectionLike,
  VisualNetworkNodeLike,
} from '../browser-entry.types';

type TemporalModuleDescriptor = ReturnType<
  Network['describeTemporalStructure']
>['recurrentModules'][number];

/**
 * Semantic annotation for one hidden column in the browser network view.
 *
 * These annotations power recurrent-role guide chips such as “input gate” or
 * “IN t-1”, making recurrent presets readable without flattening them into one
 * anonymous hidden shelf.
 */
export interface NetworkHiddenColumnAnnotation {
  label: string;
  labelLines: readonly string[];
  tooltipHeading: string;
  tooltipBodyParagraphs: readonly string[];
  backgroundColor: string;
  nodeIndices: number[];
}

type HiddenColumnTooltipContent = {
  heading: string;
  bodyParagraphs: readonly string[];
};

type TopologyEdge = readonly [fromNodeIndex: number, toNodeIndex: number];

/**
 * Full topology plan for the browser network view.
 *
 * The plan preserves the original layer-array input used by layout helpers and
 * adds semantic hidden-column annotations for recurrent-aware overlays.
 */
export interface NetworkVisualizationTopologyPlan {
  networkLayers: VisualNetworkNodeLike[][];
  hiddenColumnAnnotations: NetworkHiddenColumnAnnotation[];
}

/**
 * Topology resolution helpers for the browser network view.
 *
 * These helpers answer a key visualization question: how should the current
 * network be partitioned into ordered layers so layout and architecture labels
 * stay meaningful even when some metadata is missing?
 */

/**
 * Resolves layered node groups for network-view layout and rendering.
 *
 * @param network - Runtime network instance.
 * @param inputSize - Input count fallback.
 * @param outputSize - Output count fallback.
 * @returns Layered nodes for rendering.
 */
export function resolveNetworkVisualizationLayers(
  network: Network | undefined,
  inputSize: number,
  outputSize: number,
): VisualNetworkNodeLike[][] {
  return resolveNetworkVisualizationTopologyPlan(network, inputSize, outputSize)
    .networkLayers;
}

/**
 * Resolves the full topology plan for browser layout and recurrent guides.
 *
 * @param network - Runtime network instance.
 * @param inputSize - Input count fallback.
 * @param outputSize - Output count fallback.
 * @returns Layered nodes plus semantic hidden-column annotations.
 */
export function resolveNetworkVisualizationTopologyPlan(
  network: Network | undefined,
  inputSize: number,
  outputSize: number,
): NetworkVisualizationTopologyPlan {
  // Step 1: Build fallback layers when no runtime network is available.
  if (!network) {
    return {
      networkLayers: [
        Array.from({ length: inputSize }, (_unusedValue, inputNodeIndex) => ({
          index: inputNodeIndex,
          type: 'input',
          bias: 0,
          activation: 0,
        })),
        Array.from({ length: outputSize }, (_unusedValue, outputNodeIndex) => ({
          index: inputSize + outputNodeIndex,
          type: 'output',
          bias: 0,
          activation: 0,
        })),
      ],
      hiddenColumnAnnotations: [],
    };
  }

  // Step 2: Normalize runtime node records into the view-layer shape.
  const runtimeNodes = (
    (network.nodes ?? []) as Array<{
      index?: number;
      type?: string;
      bias?: number;
      activation?: number;
      geneId?: number;
      layer?: number;
    }>
  ).map((runtimeNode, fallbackNodeIndex) => ({
    index:
      typeof runtimeNode.index === 'number'
        ? runtimeNode.index
        : fallbackNodeIndex,
    type: runtimeNode.type ?? 'hidden',
    bias: runtimeNode.bias ?? 0,
    activation: runtimeNode.activation ?? 0,
    geneId: runtimeNode.geneId,
    layer: runtimeNode.layer,
  }));

  // Step 3: Resolve input, hidden, and output partitions.
  const inputAndConstantNodes = runtimeNodes
    .filter(
      (runtimeNode) =>
        runtimeNode.type === 'input' || runtimeNode.type === 'constant',
    )
    .toSorted((leftNode, rightNode) => leftNode.index - rightNode.index);
  const outputNodes = runtimeNodes
    .filter((runtimeNode) => runtimeNode.type === 'output')
    .toSorted((leftNode, rightNode) => leftNode.index - rightNode.index);
  const hiddenNodes = runtimeNodes.filter(
    (runtimeNode) =>
      runtimeNode.type !== 'input' &&
      runtimeNode.type !== 'constant' &&
      runtimeNode.type !== 'output',
  );

  // Step 4: Prefer explicit recurrent-module structure when it is available.
  const structuredTemporalTopologyPlan = resolveStructuredTemporalTopologyPlan(
    network,
    runtimeNodes,
    inputAndConstantNodes,
    hiddenNodes,
    outputNodes,
  );
  if (structuredTemporalTopologyPlan) {
    return structuredTemporalTopologyPlan;
  }

  // Step 5: Prefer layer metadata, then fall back to topology-derived depth.
  const hiddenLayersByMetadata = groupHiddenNodesByLayerMetadata(hiddenNodes);
  const hiddenLayers =
    hiddenLayersByMetadata.length > 0
      ? hiddenLayersByMetadata
      : groupHiddenNodesByTopology(network, runtimeNodes, hiddenNodes);
  const layeredNodes = [
    inputAndConstantNodes,
    ...hiddenLayers,
    outputNodes,
  ].filter((layerNodes) => layerNodes.length > 0);

  return {
    networkLayers: layeredNodes.length > 0 ? layeredNodes : [runtimeNodes],
    hiddenColumnAnnotations: [],
  };
}

function resolveStructuredTemporalTopologyPlan(
  network: Network,
  runtimeNodes: VisualNetworkNodeLike[],
  inputNodes: VisualNetworkNodeLike[],
  hiddenNodes: VisualNetworkNodeLike[],
  outputNodes: VisualNetworkNodeLike[],
): NetworkVisualizationTopologyPlan | undefined {
  const temporalStructure = network.describeTemporalStructure();
  if (temporalStructure.recurrentModules.length === 0) {
    return undefined;
  }

  const nodeByGeneId = createNodeByGeneIdMap(runtimeNodes);
  const orderedRecurrentModules = [
    ...temporalStructure.recurrentModules,
  ].toSorted(
    (leftModule, rightModule) =>
      resolveModuleOrderValue(leftModule) -
      resolveModuleOrderValue(rightModule),
  );
  const moduleOwnedGeneIds = new Set(
    orderedRecurrentModules.flatMap((recurrentModule) =>
      Object.values(recurrentModule.nodeGeneIdsByRole).flat(),
    ),
  );
  const remainingHiddenNodes = hiddenNodes.filter(
    (hiddenNode) => !moduleOwnedGeneIds.has(hiddenNode.geneId ?? Number.NaN),
  );
  const remainingHiddenLayers = groupHiddenNodesByTopology(
    network,
    runtimeNodes,
    remainingHiddenNodes,
    moduleOwnedGeneIds,
  );
  const recurrentKinds = [
    ...new Set(
      orderedRecurrentModules.map((recurrentModule) => recurrentModule.kind),
    ),
  ];

  if (recurrentKinds.length === 1 && recurrentKinds[0] === 'narx-memory') {
    return resolveNarxTemporalTopologyPlan(
      orderedRecurrentModules,
      nodeByGeneId,
      inputNodes,
      remainingHiddenLayers,
      outputNodes,
    );
  }

  return resolveGenericTemporalTopologyPlan(
    orderedRecurrentModules,
    nodeByGeneId,
    inputNodes,
    remainingHiddenLayers,
    outputNodes,
  );
}

function resolveNarxTemporalTopologyPlan(
  orderedRecurrentModules: readonly TemporalModuleDescriptor[],
  nodeByGeneId: Map<number, VisualNetworkNodeLike>,
  inputNodes: VisualNetworkNodeLike[],
  remainingHiddenLayers: VisualNetworkNodeLike[][],
  outputNodes: VisualNetworkNodeLike[],
): NetworkVisualizationTopologyPlan {
  const inputDelayModule = orderedRecurrentModules.find(
    (recurrentModule) => recurrentModule.moduleLabel === 'input',
  );
  const outputDelayModule = orderedRecurrentModules.find(
    (recurrentModule) => recurrentModule.moduleLabel === 'output',
  );
  const inputDelayColumns = inputDelayModule
    ? resolveModuleRoleColumns(inputDelayModule, nodeByGeneId, 0)
    : { layers: [], annotations: [] };
  const outputDelayColumns = outputDelayModule
    ? resolveModuleRoleColumns(outputDelayModule, nodeByGeneId, 3)
    : { layers: [], annotations: [] };
  const denseHiddenColumns = resolveAuxiliaryHiddenColumns(
    remainingHiddenLayers,
    'HIDDEN',
    5,
  );

  return {
    networkLayers: [
      inputNodes,
      ...inputDelayColumns.layers,
      ...denseHiddenColumns.layers,
      ...outputDelayColumns.layers,
      outputNodes,
    ].filter((layerNodes) => layerNodes.length > 0),
    hiddenColumnAnnotations: [
      ...inputDelayColumns.annotations,
      ...denseHiddenColumns.annotations,
      ...outputDelayColumns.annotations,
    ],
  };
}

function resolveGenericTemporalTopologyPlan(
  orderedRecurrentModules: readonly TemporalModuleDescriptor[],
  nodeByGeneId: Map<number, VisualNetworkNodeLike>,
  inputNodes: VisualNetworkNodeLike[],
  remainingHiddenLayers: VisualNetworkNodeLike[][],
  outputNodes: VisualNetworkNodeLike[],
): NetworkVisualizationTopologyPlan {
  const recurrentRoleColumns = orderedRecurrentModules.map(
    (recurrentModule, recurrentModuleIndex) =>
      resolveModuleRoleColumns(
        recurrentModule,
        nodeByGeneId,
        recurrentModuleIndex,
      ),
  );
  const auxiliaryHiddenColumns = resolveAuxiliaryHiddenColumns(
    remainingHiddenLayers,
    'AUX',
    orderedRecurrentModules.length + 2,
  );

  return {
    networkLayers: [
      inputNodes,
      ...recurrentRoleColumns.flatMap(
        (recurrentRoleColumn) => recurrentRoleColumn.layers,
      ),
      ...auxiliaryHiddenColumns.layers,
      outputNodes,
    ].filter((layerNodes) => layerNodes.length > 0),
    hiddenColumnAnnotations: [
      ...recurrentRoleColumns.flatMap(
        (recurrentRoleColumn) => recurrentRoleColumn.annotations,
      ),
      ...auxiliaryHiddenColumns.annotations,
    ],
  };
}

function resolveModuleRoleColumns(
  recurrentModule: TemporalModuleDescriptor,
  nodeByGeneId: Map<number, VisualNetworkNodeLike>,
  colorOffset: number,
): {
  layers: VisualNetworkNodeLike[][];
  annotations: NetworkHiddenColumnAnnotation[];
} {
  const orderedRoleNames = resolveOrderedRoleNames(recurrentModule);
  const resolvedColumns = orderedRoleNames.flatMap((roleName, roleIndex) => {
    const roleNodeGeneIds = recurrentModule.nodeGeneIdsByRole[roleName] ?? [];
    const roleNodes = roleNodeGeneIds
      .map((geneId) => nodeByGeneId.get(geneId))
      .filter((roleNode): roleNode is VisualNetworkNodeLike => roleNode != null)
      .toSorted((leftNode, rightNode) => leftNode.index - rightNode.index);
    if (roleNodes.length === 0) {
      return [];
    }

    const hiddenColumnTooltip = resolveHiddenColumnTooltip(
      recurrentModule.kind,
      roleName,
      recurrentModule,
    );

    return [
      {
        layer: roleNodes,
        annotation: {
          label: resolveRoleLabel(
            recurrentModule.kind,
            roleName,
            recurrentModule,
          ),
          labelLines: resolveRoleLabelLines(
            recurrentModule.kind,
            roleName,
            recurrentModule,
          ),
          tooltipHeading: hiddenColumnTooltip.heading,
          tooltipBodyParagraphs: hiddenColumnTooltip.bodyParagraphs,
          backgroundColor:
            FLAPPY_LIGHT_NEON_RAMP[
              (colorOffset + roleIndex) % FLAPPY_LIGHT_NEON_RAMP.length
            ] ??
            FLAPPY_LIGHT_NEON_RAMP[0] ??
            '#7fe8ff',
          nodeIndices: roleNodes.map((roleNode) => roleNode.index),
        },
      },
    ];
  });

  return {
    layers: resolvedColumns.map((resolvedColumn) => resolvedColumn.layer),
    annotations: resolvedColumns.map(
      (resolvedColumn) => resolvedColumn.annotation,
    ),
  };
}

function resolveAuxiliaryHiddenColumns(
  hiddenLayers: VisualNetworkNodeLike[][],
  labelPrefix: string,
  colorOffset: number,
): {
  layers: VisualNetworkNodeLike[][];
  annotations: NetworkHiddenColumnAnnotation[];
} {
  return {
    layers: hiddenLayers,
    annotations: hiddenLayers.map((hiddenLayer, hiddenLayerIndex) => {
      const hiddenColumnTooltip = resolveAuxiliaryHiddenTooltip(
        labelPrefix,
        hiddenLayerIndex,
      );

      return {
        label: `${labelPrefix} ${hiddenLayerIndex + 1}`,
        labelLines: [labelPrefix, String(hiddenLayerIndex + 1)],
        tooltipHeading: hiddenColumnTooltip.heading,
        tooltipBodyParagraphs: hiddenColumnTooltip.bodyParagraphs,
        backgroundColor:
          FLAPPY_LIGHT_NEON_RAMP[
            (colorOffset + hiddenLayerIndex) % FLAPPY_LIGHT_NEON_RAMP.length
          ] ??
          FLAPPY_LIGHT_NEON_RAMP[0] ??
          '#7fe8ff',
        nodeIndices: hiddenLayer.map((hiddenNode) => hiddenNode.index),
      };
    }),
  };
}

function createNodeByGeneIdMap(
  runtimeNodes: VisualNetworkNodeLike[],
): Map<number, VisualNetworkNodeLike> {
  return new Map(
    runtimeNodes
      .filter((runtimeNode) => Number.isFinite(runtimeNode.geneId))
      .map((runtimeNode) => [runtimeNode.geneId as number, runtimeNode]),
  );
}

function resolveModuleOrderValue(
  recurrentModule: TemporalModuleDescriptor,
): number {
  const orderedGeneIds = Object.values(recurrentModule.nodeGeneIdsByRole)
    .flat()
    .filter((geneId): geneId is number => Number.isFinite(geneId));
  return orderedGeneIds.length > 0
    ? Math.min(...orderedGeneIds)
    : Number.MAX_SAFE_INTEGER;
}

function resolveOrderedRoleNames(
  recurrentModule: TemporalModuleDescriptor,
): string[] {
  if (recurrentModule.kind === 'lstm') {
    return [
      'inputGate',
      'forgetGate',
      'memoryCell',
      'outputGate',
      'outputBlock',
    ].filter((roleName) => roleName in recurrentModule.nodeGeneIdsByRole);
  }

  if (recurrentModule.kind === 'gru') {
    return [
      'updateGate',
      'inverseUpdateGate',
      'resetGate',
      'memoryCell',
      'output',
      'previousOutput',
    ].filter((roleName) => roleName in recurrentModule.nodeGeneIdsByRole);
  }

  return Object.keys(recurrentModule.nodeGeneIdsByRole).toSorted(
    (leftRoleName, rightRoleName) =>
      resolveNarxDelayRoleIndex(leftRoleName) -
      resolveNarxDelayRoleIndex(rightRoleName),
  );
}

function resolveRoleLabel(
  recurrentKind: TemporalModuleDescriptor['kind'],
  roleName: string,
  recurrentModule: TemporalModuleDescriptor,
): string {
  if (recurrentKind === 'narx-memory') {
    const delayIndex = resolveNarxDelayRoleIndex(roleName) + 1;
    const modulePrefix =
      recurrentModule.moduleLabel === 'output' ? 'OUT' : 'IN';
    return `${modulePrefix} t-${delayIndex}`;
  }

  if (recurrentKind === 'lstm') {
    return (
      (
        {
          inputGate: 'INPUT GATE',
          forgetGate: 'FORGET GATE',
          memoryCell: 'MEMORY CELL',
          outputGate: 'OUTPUT GATE',
          outputBlock: 'OUTPUT BLOCK',
        } as Record<string, string>
      )[roleName] ?? roleName.toUpperCase()
    );
  }

  return (
    (
      {
        updateGate: 'UPDATE GATE',
        inverseUpdateGate: 'INVERSE UPDATE',
        resetGate: 'RESET GATE',
        memoryCell: 'MEMORY CELL',
        output: 'OUTPUT MIX',
        previousOutput: 'PREV OUTPUT',
      } as Record<string, string>
    )[roleName] ?? roleName.toUpperCase()
  );
}

function resolveRoleLabelLines(
  recurrentKind: TemporalModuleDescriptor['kind'],
  roleName: string,
  recurrentModule: TemporalModuleDescriptor,
): readonly string[] {
  if (recurrentKind === 'narx-memory') {
    const delayIndex = resolveNarxDelayRoleIndex(roleName) + 1;
    const modulePrefix =
      recurrentModule.moduleLabel === 'output' ? 'OUT' : 'IN';
    return [modulePrefix, `t-${delayIndex}`];
  }

  if (recurrentKind === 'lstm') {
    return (
      (
        {
          inputGate: ['INPUT', 'GATE'],
          forgetGate: ['FORGET', 'GATE'],
          memoryCell: ['MEMORY', 'CELL'],
          outputGate: ['OUTPUT', 'GATE'],
          outputBlock: ['OUTPUT', 'BLOCK'],
        } as Record<string, readonly string[]>
      )[roleName] ?? [
        resolveRoleLabel(recurrentKind, roleName, recurrentModule),
      ]
    );
  }

  return (
    (
      {
        updateGate: ['UPDATE', 'GATE'],
        inverseUpdateGate: ['INV', 'UPDATE'],
        resetGate: ['RESET', 'GATE'],
        memoryCell: ['MEMORY', 'CELL'],
        output: ['OUTPUT', 'MIX'],
        previousOutput: ['PREV', 'OUTPUT'],
      } as Record<string, readonly string[]>
    )[roleName] ?? [resolveRoleLabel(recurrentKind, roleName, recurrentModule)]
  );
}

function resolveHiddenColumnTooltip(
  recurrentKind: TemporalModuleDescriptor['kind'],
  roleName: string,
  recurrentModule: TemporalModuleDescriptor,
): HiddenColumnTooltipContent {
  if (recurrentKind === 'lstm') {
    return resolveLstmRoleTooltip(roleName);
  }

  if (recurrentKind === 'gru') {
    return resolveGruRoleTooltip(roleName);
  }

  return resolveNarxRoleTooltip(roleName, recurrentModule);
}

function resolveLstmRoleTooltip(roleName: string): HiddenColumnTooltipContent {
  return (
    (
      {
        inputGate: {
          heading: 'LSTM Input Gate',
          bodyParagraphs: [
            'The input gate decides how much new evidence is allowed to write into the cell state on this step.',
            'Open it wider and the block learns quickly from the present input; close it and the cell protects older memory.',
          ],
        },
        forgetGate: {
          heading: 'LSTM Forget Gate',
          bodyParagraphs: [
            'The forget gate decides how much of the previous cell state survives into the next step.',
            "It is the LSTM's erase control, letting the network drop stale context instead of carrying every old signal forever.",
          ],
        },
        memoryCell: {
          heading: 'LSTM Memory Cell',
          bodyParagraphs: [
            'The memory cell is the long-lived state lane that carries accumulated context across time.',
            'Because the gates regulate what gets written, preserved, and exposed, this shelf can remember patterns longer than a plain recurrent loop.',
          ],
        },
        outputGate: {
          heading: 'LSTM Output Gate',
          bodyParagraphs: [
            'The output gate decides how much of the cell state is revealed to the rest of the network right now.',
            'That separation lets the block keep useful memory internally without broadcasting all of it at every timestep.',
          ],
        },
        outputBlock: {
          heading: 'LSTM Output Block',
          bodyParagraphs: [
            'This column is the exposed state emitted after the memory cell has passed through the output gate.',
            'Other layers read this shelf directly, while the deeper cell memory can still keep extra context private.',
          ],
        },
      } as Record<string, HiddenColumnTooltipContent>
    )[roleName] ?? resolveGenericRecurrentRoleTooltip('LSTM role', roleName)
  );
}

function resolveGruRoleTooltip(roleName: string): HiddenColumnTooltipContent {
  return (
    (
      {
        updateGate: {
          heading: 'GRU Update Gate',
          bodyParagraphs: [
            'The update gate chooses how much of the old state survives into the next output.',
            "It is the GRU's main carry-versus-overwrite dial, blending memory retention with new evidence.",
          ],
        },
        inverseUpdateGate: {
          heading: 'GRU Inverse-Update Branch',
          bodyParagraphs: [
            'This branch is the complement of the update gate: it measures how much room is left for the new candidate state.',
            'When the carry signal stays low, this shelf grows stronger and gives fresh evidence a larger share of the final mix.',
          ],
        },
        resetGate: {
          heading: 'GRU Reset Gate',
          bodyParagraphs: [
            'The reset gate decides how much of the previous state the candidate generator is allowed to inspect.',
            'Closing it makes the candidate behave more like a fresh local reaction; opening it lets older context shape the new proposal.',
          ],
        },
        memoryCell: {
          heading: 'GRU Candidate Memory',
          bodyParagraphs: [
            'In this GRU, the memory cell is the candidate-state workshop rather than the final state itself.',
            'It combines current input with reset-filtered history to propose what the unit would believe if it decided to refresh.',
          ],
        },
        output: {
          heading: 'GRU Output Mix',
          bodyParagraphs: [
            'The output mix is the actual recurrent state emitted by the GRU at this step.',
            'It is the weighted blend of carried old state and the new candidate, so downstream layers see the final compromise instead of the raw proposal.',
          ],
        },
        previousOutput: {
          heading: 'GRU Previous Output',
          bodyParagraphs: [
            'This shelf is the delayed state fed back from the previous timestep.',
            'It is the recurrent trace the rest of the block decides to preserve, ignore, or remix into something new.',
          ],
        },
      } as Record<string, HiddenColumnTooltipContent>
    )[roleName] ?? resolveGenericRecurrentRoleTooltip('GRU role', roleName)
  );
}

function resolveNarxRoleTooltip(
  roleName: string,
  recurrentModule: TemporalModuleDescriptor,
): HiddenColumnTooltipContent {
  const delayIndex = resolveNarxDelayRoleIndex(roleName) + 1;
  const isOutputDelay = recurrentModule.moduleLabel === 'output';

  return isOutputDelay
    ? {
        heading: `NARX Output Delay t-${delayIndex}`,
        bodyParagraphs: [
          'This shelf stores one older model output, giving the network an explicit autoregressive trace of its own recent behavior.',
          'Together with the input delays, these taps let NARX predict the next step from both what just happened and what it recently produced.',
        ],
      }
    : {
        heading: `NARX Input Delay t-${delayIndex}`,
        bodyParagraphs: [
          'This shelf stores one older external input, so the network reads recent history explicitly instead of burying it inside a gated cell.',
          'Stack several taps and NARX becomes a learned sliding window over the recent input sequence.',
        ],
      };
}

function resolveAuxiliaryHiddenTooltip(
  labelPrefix: string,
  hiddenLayerIndex: number,
): HiddenColumnTooltipContent {
  if (labelPrefix === 'HIDDEN') {
    return {
      heading: `Dense Hidden Layer ${hiddenLayerIndex + 1}`,
      bodyParagraphs: [
        'This column is not a delay shelf; it is a standard nonlinear workspace that combines the remembered taps into prediction features.',
        'In NARX-style layouts, these layers often learn interactions like trend, curvature, and correction signals across the explicit history window.',
      ],
    };
  }

  return {
    heading: `Auxiliary Hidden Column ${hiddenLayerIndex + 1}`,
    bodyParagraphs: [
      'This column is not one of the named recurrent roles; it is extra nonlinear workspace between the recurrent block and the readout.',
      'When it becomes useful, it acts like a translator that reshapes memory-bearing state into a cleaner decision signal.',
    ],
  };
}

function resolveGenericRecurrentRoleTooltip(
  familyLabel: string,
  roleName: string,
): HiddenColumnTooltipContent {
  return {
    heading: familyLabel,
    bodyParagraphs: [
      `This column belongs to the ${familyLabel.toLowerCase()} boundary but does not have a hand-written teaching card yet.`,
      `Read it as the ${roleName} sub-role that helps the recurrent block decide what to keep, transform, or expose over time.`,
    ],
  };
}

function resolveNarxDelayRoleIndex(roleName: string): number {
  const matchedDelayIndex = /^delayStep(\d+)$/.exec(roleName)?.[1];
  return typeof matchedDelayIndex === 'string'
    ? Number.parseInt(matchedDelayIndex, 10)
    : Number.MAX_SAFE_INTEGER;
}

function groupHiddenNodesByLayerMetadata(
  hiddenNodes: VisualNetworkNodeLike[],
): VisualNetworkNodeLike[][] {
  const hiddenNodesWithLayer = hiddenNodes.filter(
    (hiddenNode) => typeof hiddenNode.layer === 'number',
  );
  if (hiddenNodesWithLayer.length === 0) {
    return [];
  }

  const nodesByLayer = new Map<number, VisualNetworkNodeLike[]>();
  hiddenNodesWithLayer.forEach((hiddenNode) => {
    const layerIndex = hiddenNode.layer as number;
    const existingLayerNodes = nodesByLayer.get(layerIndex) ?? [];
    existingLayerNodes.push(hiddenNode);
    nodesByLayer.set(layerIndex, existingLayerNodes);
  });

  return [...nodesByLayer.entries()]
    .toSorted(
      (leftLayerEntry, rightLayerEntry) =>
        leftLayerEntry[0] - rightLayerEntry[0],
    )
    .map((layerEntry) =>
      layerEntry[1].toSorted(
        (leftNode, rightNode) => leftNode.index - rightNode.index,
      ),
    );
}

function groupHiddenNodesByTopology(
  network: Network,
  runtimeNodes: VisualNetworkNodeLike[],
  hiddenNodes: VisualNetworkNodeLike[],
  excludedGeneIds: ReadonlySet<number> = new Set<number>(),
): VisualNetworkNodeLike[][] {
  if (hiddenNodes.length === 0) {
    return [];
  }

  const filteredRuntimeNodes = runtimeNodes.filter(
    (runtimeNode) => !excludedGeneIds.has(runtimeNode.geneId ?? Number.NaN),
  );
  const filteredRuntimeNodeIndices = new Set(
    filteredRuntimeNodes.map((runtimeNode) => runtimeNode.index),
  );
  const filteredRuntimeConnections = (
    (network.connections ?? []) as VisualNetworkConnectionLike[]
  ).filter((runtimeConnection) => {
    const fromNodeIndex = runtimeConnection.from?.index;
    const toNodeIndex = runtimeConnection.to?.index;
    return (
      typeof fromNodeIndex === 'number' &&
      typeof toNodeIndex === 'number' &&
      filteredRuntimeNodeIndices.has(fromNodeIndex) &&
      filteredRuntimeNodeIndices.has(toNodeIndex)
    );
  });

  const hiddenDepthByNodeIndex = resolveHiddenNodeDepthByTopology(
    filteredRuntimeNodes,
    filteredRuntimeConnections,
  );
  const nodesByDepth = new Map<number, VisualNetworkNodeLike[]>();

  hiddenNodes.forEach((hiddenNode) => {
    const depth = hiddenDepthByNodeIndex.get(hiddenNode.index) ?? 1;
    const existingDepthNodes = nodesByDepth.get(depth) ?? [];
    existingDepthNodes.push(hiddenNode);
    nodesByDepth.set(depth, existingDepthNodes);
  });

  return [...nodesByDepth.entries()]
    .toSorted(
      (leftDepthEntry, rightDepthEntry) =>
        leftDepthEntry[0] - rightDepthEntry[0],
    )
    .map((depthEntry) =>
      depthEntry[1].toSorted(
        (leftNode, rightNode) => leftNode.index - rightNode.index,
      ),
    );
}

function resolveHiddenNodeDepthByTopology(
  runtimeNodes: VisualNetworkNodeLike[],
  runtimeConnections: VisualNetworkConnectionLike[],
): Map<number, number> {
  const nodeByIndex = new Map<number, VisualNetworkNodeLike>(
    runtimeNodes.map((runtimeNode) => [runtimeNode.index, runtimeNode]),
  );
  const topologyEdges = collectEnabledTopologyEdges(
    nodeByIndex,
    runtimeConnections,
  );
  const topologyComponents = resolveStronglyConnectedTopologyComponents(
    [...nodeByIndex.keys()],
    topologyEdges,
  );
  const depthByComponentIndex = resolveTopologyComponentDepths(
    nodeByIndex,
    topologyEdges,
    topologyComponents,
  );

  return new Map<number, number>(
    runtimeNodes
      .filter((runtimeNode) => runtimeNode.type === 'hidden')
      .map((runtimeNode) => [
        runtimeNode.index,
        depthByComponentIndex.get(
          topologyComponents.componentIndexByNodeIndex.get(runtimeNode.index) ??
            Number.NaN,
        ) ?? 1,
      ]),
  );
}

function collectEnabledTopologyEdges(
  nodeByIndex: ReadonlyMap<number, VisualNetworkNodeLike>,
  runtimeConnections: readonly VisualNetworkConnectionLike[],
): TopologyEdge[] {
  return runtimeConnections.flatMap((runtimeConnection) => {
    if (runtimeConnection.enabled === false) {
      return [];
    }

    const fromNodeIndex = runtimeConnection.from?.index;
    const toNodeIndex = runtimeConnection.to?.index;
    if (
      typeof fromNodeIndex !== 'number' ||
      typeof toNodeIndex !== 'number' ||
      !nodeByIndex.has(fromNodeIndex) ||
      !nodeByIndex.has(toNodeIndex) ||
      fromNodeIndex === toNodeIndex
    ) {
      return [];
    }

    return [[fromNodeIndex, toNodeIndex] as TopologyEdge];
  });
}

function resolveStronglyConnectedTopologyComponents(
  nodeIndices: readonly number[],
  topologyEdges: readonly TopologyEdge[],
): {
  componentIndexByNodeIndex: Map<number, number>;
  components: number[][];
} {
  const outgoingTargetsByNode = new Map<number, number[]>();
  const componentIndexByNodeIndex = new Map<number, number>();
  const discoveryIndexByNode = new Map<number, number>();
  const lowlinkByNode = new Map<number, number>();
  const nodeIndicesOnStack = new Set<number>();
  const nodeIndexStack: number[] = [];
  const components: number[][] = [];
  let nextDiscoveryIndex = 0;

  nodeIndices.forEach((nodeIndex) => {
    outgoingTargetsByNode.set(nodeIndex, []);
  });
  topologyEdges.forEach(([fromNodeIndex, toNodeIndex]) => {
    const outgoingTargets = outgoingTargetsByNode.get(fromNodeIndex) ?? [];
    outgoingTargets.push(toNodeIndex);
    outgoingTargetsByNode.set(fromNodeIndex, outgoingTargets);
  });

  nodeIndices.forEach((nodeIndex) => {
    if (!discoveryIndexByNode.has(nodeIndex)) {
      visitNode(nodeIndex);
    }
  });

  return {
    componentIndexByNodeIndex,
    components,
  };

  function visitNode(nodeIndex: number): void {
    discoveryIndexByNode.set(nodeIndex, nextDiscoveryIndex);
    lowlinkByNode.set(nodeIndex, nextDiscoveryIndex);
    nextDiscoveryIndex += 1;
    nodeIndexStack.push(nodeIndex);
    nodeIndicesOnStack.add(nodeIndex);

    const outgoingTargets = outgoingTargetsByNode.get(nodeIndex) ?? [];
    outgoingTargets.forEach((targetNodeIndex) => {
      if (!discoveryIndexByNode.has(targetNodeIndex)) {
        visitNode(targetNodeIndex);
        lowlinkByNode.set(
          nodeIndex,
          Math.min(
            lowlinkByNode.get(nodeIndex) ?? Number.MAX_SAFE_INTEGER,
            lowlinkByNode.get(targetNodeIndex) ?? Number.MAX_SAFE_INTEGER,
          ),
        );
        return;
      }

      if (nodeIndicesOnStack.has(targetNodeIndex)) {
        lowlinkByNode.set(
          nodeIndex,
          Math.min(
            lowlinkByNode.get(nodeIndex) ?? Number.MAX_SAFE_INTEGER,
            discoveryIndexByNode.get(targetNodeIndex) ??
              Number.MAX_SAFE_INTEGER,
          ),
        );
      }
    });

    if (
      (lowlinkByNode.get(nodeIndex) ?? Number.NaN) !==
      (discoveryIndexByNode.get(nodeIndex) ?? Number.NaN)
    ) {
      return;
    }

    const componentNodes: number[] = [];
    while (nodeIndexStack.length > 0) {
      const stackNodeIndex = nodeIndexStack.pop();
      if (typeof stackNodeIndex !== 'number') {
        break;
      }

      nodeIndicesOnStack.delete(stackNodeIndex);
      componentIndexByNodeIndex.set(stackNodeIndex, components.length);
      componentNodes.push(stackNodeIndex);
      if (stackNodeIndex === nodeIndex) {
        break;
      }
    }

    components.push(
      componentNodes.toSorted(
        (leftNodeIndex, rightNodeIndex) => leftNodeIndex - rightNodeIndex,
      ),
    );
  }
}

function resolveTopologyComponentDepths(
  nodeByIndex: ReadonlyMap<number, VisualNetworkNodeLike>,
  topologyEdges: readonly TopologyEdge[],
  topologyComponents: {
    componentIndexByNodeIndex: ReadonlyMap<number, number>;
    components: readonly number[][];
  },
): Map<number, number> {
  const outgoingTargetsByComponentIndex = new Map<number, number[]>();
  const incomingEdgeCountByComponentIndex = new Map<number, number>();
  const componentDepthByIndex = new Map<number, number>();
  const componentEdgeKeys = new Set<string>();

  topologyComponents.components.forEach(
    (componentNodeIndices, componentIndex) => {
      outgoingTargetsByComponentIndex.set(componentIndex, []);
      incomingEdgeCountByComponentIndex.set(componentIndex, 0);
      componentDepthByIndex.set(
        componentIndex,
        componentNodeIndices.some((nodeIndex) => {
          const nodeType = nodeByIndex.get(nodeIndex)?.type;
          return nodeType === 'input' || nodeType === 'constant';
        })
          ? 0
          : 1,
      );
    },
  );

  topologyEdges.forEach(([fromNodeIndex, toNodeIndex]) => {
    const fromComponentIndex =
      topologyComponents.componentIndexByNodeIndex.get(fromNodeIndex);
    const toComponentIndex =
      topologyComponents.componentIndexByNodeIndex.get(toNodeIndex);
    if (
      typeof fromComponentIndex !== 'number' ||
      typeof toComponentIndex !== 'number' ||
      fromComponentIndex === toComponentIndex
    ) {
      return;
    }

    const componentEdgeKey = `${fromComponentIndex}:${toComponentIndex}`;
    if (componentEdgeKeys.has(componentEdgeKey)) {
      return;
    }

    componentEdgeKeys.add(componentEdgeKey);
    const outgoingTargets =
      outgoingTargetsByComponentIndex.get(fromComponentIndex) ?? [];
    outgoingTargets.push(toComponentIndex);
    outgoingTargetsByComponentIndex.set(fromComponentIndex, outgoingTargets);
    incomingEdgeCountByComponentIndex.set(
      toComponentIndex,
      (incomingEdgeCountByComponentIndex.get(toComponentIndex) ?? 0) + 1,
    );
  });

  const componentQueue = [...incomingEdgeCountByComponentIndex.entries()]
    .filter((incomingEntry) => incomingEntry[1] === 0)
    .map((incomingEntry) => incomingEntry[0]);

  while (componentQueue.length > 0) {
    const currentComponentIndex = componentQueue.shift();
    if (typeof currentComponentIndex !== 'number') {
      continue;
    }

    const currentComponentDepth =
      componentDepthByIndex.get(currentComponentIndex) ?? 0;
    const outgoingTargets =
      outgoingTargetsByComponentIndex.get(currentComponentIndex) ?? [];
    outgoingTargets.forEach((targetComponentIndex) => {
      componentDepthByIndex.set(
        targetComponentIndex,
        Math.max(
          componentDepthByIndex.get(targetComponentIndex) ?? 1,
          currentComponentDepth + 1,
        ),
      );

      const remainingIncomingCount =
        (incomingEdgeCountByComponentIndex.get(targetComponentIndex) ?? 0) - 1;
      incomingEdgeCountByComponentIndex.set(
        targetComponentIndex,
        remainingIncomingCount,
      );
      if (remainingIncomingCount === 0) {
        componentQueue.push(targetComponentIndex);
      }
    });
  }

  return componentDepthByIndex;
}
