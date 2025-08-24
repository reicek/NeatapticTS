/**
 * Network Visualization - Handles neural network visualization for terminal display
 *
 * This module contains functions for visualizing neural networks in the terminal,
 * providing an ASCII representation of the network architecture and activation values.
 *
 * These visualizations help in understanding:
 * - Network architecture (inputs, hidden layers, outputs)
 * - Activation patterns during maze solving
 * - Connection structure between layers
 */

import { INetwork } from './interfaces'; // Added INetwork import
import { MazeUtils } from './mazeUtils';
import { colors } from './colors';
import { IVisualizationNode, IVisualizationConnection } from './interfaces';

// Ambient declaration fallback: some TS lib bundlings in test build may not surface ES2023 array methods.
// This doesn't polyfill at runtime (Node 20+ already supports them); it merely satisfies the type checker.
interface Array<T> {
  toSorted?(compareFn?: (a: T, b: T) => number): T[];
}

/**
 * NetworkVisualization
 *
 * Utility class implementing ASCII visualizations of neural networks used by
 * the ASCII maze demo. Methods focus on producing compact, colorized strings
 * for terminal display. Performance-sensitive areas avoid temporary
 * allocations when possible (e.g., building small windowed arrays).
 */
/**
 * Provides ES2023-friendly, terminal-oriented ASCII visualizations for neural network
 * architectures used in the ASCII maze demo. Emphasizes low allocation patterns and
 * color-enhanced readability for activations and structural summaries.
 *
 * Design goals:
 * 1. Low GC pressure during rapid re-render loops (reuse scratch arrays/strings).
 * 2. Clear, color-coded activation ranges (TRON-inspired palette).
 * 3. Graceful condensation of very large hidden layers via averaged activation groups.
 * 4. Deterministic ordering using stable, non-mutating sort helpers (ES2023 toSorted fallback).
 *
 * Key public surface:
 * - {@link NetworkVisualization.visualizeNetworkSummary} produces a multi-line string summary.
 */
export class NetworkVisualization {
  // Internal layout constants (private)
  static readonly #ARROW = '  ──▶  ';
  static readonly #ARROW_WIDTH = NetworkVisualization.#ARROW.length;
  static readonly #TOTAL_WIDTH = 150; // Overall visualization width
  /** Activation range buckets (ordered, positive to negative). */
  static readonly #ACTIVATION_RANGES = [
    { min: 2.0, max: Infinity, label: 'v-high+' },
    { min: 1.0, max: 2.0, label: 'high+' },
    { min: 0.5, max: 1.0, label: 'mid+' },
    { min: 0.1, max: 0.5, label: 'low+' },
    { min: -0.1, max: 0.1, label: 'zero±' },
    { min: -0.5, max: -0.1, label: 'low-' },
    { min: -1.0, max: -0.5, label: 'mid-' },
    { min: -2.0, max: -1.0, label: 'high-' },
    { min: -Infinity, max: -2.0, label: 'v-high-' },
  ] as const;
  /** Scratch array for connection counts (reused / grown). @remarks Non-reentrant. */
  static #ConnectionCountsScratch: Int32Array = new Int32Array(16);
  static #ConnectionCountsLen = 0;
  /** Scratch list for building output rows before join. @remarks Non-reentrant. */
  static #ScratchRows: string[] = [];
  /** Scratch list for header construction. */
  static #ScratchHeaderParts: string[] = [];
  /**
   * Pads a string to a specific width with alignment options.
   *
   * @param str - String to pad.
   * @param width - Target width for the string.
   * @param padChar - Character to use for padding (default: space).
   * @param align - Alignment option ('left', 'center', or 'right').
   * @returns Padded string of specified width with chosen alignment.
   */
  static pad(
    str: string,
    width: number,
    padChar: string = ' ',
    align: 'left' | 'center' | 'right' = 'center'
  ): string {
    str = str ?? '';
    const len = str.replace(/\x1b\[[0-9;]*m/g, '').length; // Account for ANSI color codes
    if (len >= width) return str;

    const padLen = width - len;
    if (align === 'left') return str + padChar.repeat(padLen);
    if (align === 'right') return padChar.repeat(padLen) + str;

    const left = Math.floor(padLen / 2);
    const right = padLen - left;
    return padChar.repeat(left) + str + padChar.repeat(right);
  }

  /**
   * Gets activation value from a node, with safety checks.
   * For output nodes, ensures values are properly clamped between 0 and 1.
   *
   * @param node - Neural network node object.
   * @returns Cleaned and normalized activation value.
   */
  static #getNodeValue(node: any): number {
    if (
      typeof node.activation === 'number' &&
      isFinite(node.activation) &&
      !isNaN(node.activation)
    ) {
      // For output nodes, clamp between 0 and 1 for proper display
      if (node.type === 'output') {
        return Math.max(0, Math.min(1, node.activation));
      }
      // For other node types, allow a wider range but still cap for display
      return Math.max(-999, Math.min(999, node.activation));
    }
    return 0;
  }

  /**
   * Gets the appropriate color for an activation value based on its range.
   * Uses a TRON-inspired color palette for activation values.
   *
   * @param value - Activation value to colorize.
   * @returns ANSI color code for the value.
   */
  static #getActivationColor(value: number): string {
    // Use TRON-inspired color palette for activation values
    if (value >= 2.0) return colors.bgOrangeNeon + colors.bright; // Very high positive
    if (value >= 1.0) return colors.orangeNeon; // High positive
    if (value >= 0.5) return colors.cyanNeon; // Medium positive
    if (value >= 0.1) return colors.neonGreen; // Low positive
    if (value >= -0.1) return colors.whiteNeon; // Near zero
    if (value >= -0.5) return colors.blue; // Low negative
    if (value >= -1.0) return colors.blueCore; // Medium negative
    if (value >= -2.0) return colors.bgNeonAqua + colors.bright; // High negative
    return colors.bgNeonViolet + colors.neonSilver; // Very high negative
  }

  /**
   * Formats a numeric value for display with color based on its value.
   *
   * @param v - Numeric value to format.
   * @returns Colorized string representation of the value.
   */
  static #fmtColoredValue(v: number): string {
    if (typeof v !== 'number' || isNaN(v) || !isFinite(v)) return ' 0.000';

    const color = this.#getActivationColor(v);
    let formattedValue;

    formattedValue = (v >= 0 ? ' ' : '') + v.toFixed(6);

    return color + formattedValue + colors.reset;
  }

  /**
   * Format a node display with a colored symbol and its colored numeric value.
   * `extra` can include any trailing text (already colorized) and may include a leading space.
   */
  static #formatNode(
    symbolColor: string,
    symbol: string,
    node: any,
    extra?: string
  ): string {
    const value = NetworkVisualization.#getNodeValue(node);
    const fmt = NetworkVisualization.#fmtColoredValue(value);
    const sym = `${symbolColor}${symbol}${colors.reset}`;
    return `${sym}${fmt}${extra ?? ''}`;
  }

  /**
   * Groups hidden nodes into layers based on their connections.
   *
   * @param inputNodes - Array of input nodes.
   * @param hiddenNodes - Array of hidden nodes.
   * @param outputNodes - Array of output nodes.
   * @returns Array of hidden node arrays, each representing a layer.
   */
  static #groupHiddenByLayer(
    inputNodes: any[],
    hiddenNodes: any[],
    outputNodes: any[]
  ): any[][] {
    if (hiddenNodes.length === 0) return [];

    let layers: any[][] = [];
    let prevLayer = inputNodes;
    let remaining = [...hiddenNodes];

    while (remaining.length > 0) {
      const currentLayer = remaining.filter(
        (h) =>
          h.connections &&
          h.connections.in &&
          h.connections.in.length > 0 &&
          h.connections.in.every((conn: any) => prevLayer.includes(conn.from))
      );

      if (currentLayer.length === 0) {
        layers.push(remaining);
        break;
      }

      layers.push(currentLayer);
      prevLayer = currentLayer;
      remaining = remaining.filter((h) => !currentLayer.includes(h));
    }

    return layers;
  }

  /**
   * Groups nodes by their activation values to create meaningful average representations.
   * Creates more granular grouping based on activation ranges.
   *
   * @param nodes - Array of neural network nodes to group.
   * @returns Object containing groups of nodes and corresponding labels.
   */
  static #groupNodesByActivation(
    nodes: any[]
  ): {
    groups: any[][];
    labels: string[];
  } {
    // Calculate activation values once (reuse for range checks)
    const activations = nodes.map((node) =>
      NetworkVisualization.#getNodeValue(node)
    );
    /**
     * Arrays to hold groups of nodes and their labels.
     */
    const groups: any[][] = [];
    const labels: string[] = [];

    // Group nodes by predefined activation ranges
    for (const range of NetworkVisualization.#ACTIVATION_RANGES) {
      const nodesInRange = nodes.filter(
        (_, i) => activations[i] >= range.min && activations[i] < range.max
      );

      if (nodesInRange.length > 0) {
        groups.push(nodesInRange);
        labels.push(range.label);
      }
    }

    return { groups, labels };
  }

  /**
   * Prepares hidden layers for display, condensing large layers
   * to show all nodes as averages with meaningful distribution.
   *
   * @param hiddenLayers - Array of hidden layer node arrays.
   * @param maxVisiblePerLayer - Maximum number of nodes to display per layer.
   * @returns Object containing display-ready layers and metrics.
   */
  static #prepareHiddenLayersForDisplay(
    hiddenLayers: any[][],
    maxVisiblePerLayer: number = 10
  ): {
    displayLayers: any[][];
    layerDisplayCounts: number[];
    averageNodes: { [key: string]: { avgValue: number; count: number } };
  } {
    // Step 0: Fast return if no hidden layers
    /**
     * Maximum number of nodes to display per layer (rest are averaged).
     */
    const MAX_VISIBLE = maxVisiblePerLayer;

    /**
     * Stores average node info for each group.
     */
    const averageNodes: {
      [key: string]: { avgValue: number; count: number };
    } = {};

    /**
     * Arrays for display-ready layers and their display counts.
     */
    const displayLayers: any[][] = [];
    const layerDisplayCounts: number[] = [];

    hiddenLayers.forEach((layer, layerIdx) => {
      if (layer.length <= MAX_VISIBLE) {
        // If layer is small enough, show all nodes
        displayLayers.push([...layer]);
        layerDisplayCounts.push(layer.length);
      } else {
        // For large layers, show all nodes as averages to better represent distribution
        const {
          avgNodes,
          count,
        } = NetworkVisualization.#createAverageNodesForLargeLayer({
          layer,
          layerIndex: layerIdx,
          maxVisible: MAX_VISIBLE,
          averageNodesStore: averageNodes,
        });
        displayLayers.push(avgNodes);
        layerDisplayCounts.push(count);
      }
    });

    return { displayLayers, layerDisplayCounts, averageNodes };
  }

  /**
   * Create average nodes representation for a large hidden layer.
   * Decomposed from #prepareHiddenLayersForDisplay for clarity.
   */
  static #createAverageNodesForLargeLayer(params: {
    layer: any[];
    layerIndex: number;
    maxVisible: number;
    averageNodesStore: { [key: string]: { avgValue: number; count: number } };
  }): { avgNodes: any[]; count: number } {
    const { layer, layerIndex, maxVisible, averageNodesStore } = params;
    const { groups, labels } = NetworkVisualization.#groupNodesByActivation(
      layer
    );
    // If too many groups, merge using ranking strategy
    const { finalGroups, finalLabels } =
      groups.length > maxVisible
        ? NetworkVisualization.#rankMergeAndOrderGroups({
            groups,
            labels,
            maxVisible,
          })
        : { finalGroups: groups, finalLabels: labels };
    // Map groups to virtual average nodes
    const averageNodes = finalGroups.map((group, groupIndex) =>
      NetworkVisualization.#buildAverageNode({
        group,
        groupIndex,
        layerIndex,
        label: finalLabels[groupIndex],
        averageNodesStore,
      })
    );
    return { avgNodes: averageNodes, count: averageNodes.length };
  }

  /** Build a single average node descriptor from a group. */
  static #buildAverageNode(params: {
    group: any[];
    groupIndex: number;
    layerIndex: number;
    label: string;
    averageNodesStore: { [key: string]: { avgValue: number; count: number } };
  }): any {
    const { group, groupIndex, layerIndex, label, averageNodesStore } = params;
    const avgKey = `layer${layerIndex}-avg-${groupIndex}`;
    const sum = group.reduce(
      (runningTotal: number, node: any) =>
        runningTotal + NetworkVisualization.#getNodeValue(node),
      0
    );
    const avgValue = group.length ? sum / group.length : 0;
    averageNodesStore[avgKey] = { avgValue, count: group.length };
    return {
      id: -1 * (layerIndex * 1000 + groupIndex),
      uuid: avgKey,
      type: 'hidden',
      activation: avgValue,
      isAverage: true,
      avgCount: group.length,
      label,
    };
  }

  /** Rank groups by size, merge overflow into one group, then order by activation semantics. */
  static #rankMergeAndOrderGroups(params: {
    groups: any[][];
    labels: string[];
    maxVisible: number;
  }): { finalGroups: any[][]; finalLabels: string[] } {
    const { groups, labels, maxVisible } = params;
    // Pair groups with metadata
    const groupMeta = groups.map((group, index) => ({
      group,
      label: labels[index],
      size: group.length,
    }));
    // Use ES2023 toSorted via helper (non-mutating, with fallback)
    const ranked = NetworkVisualization.#safeToSorted(
      groupMeta,
      (a, b) => b.size - a.size
    );
    const cutPoint = Math.max(0, maxVisible - 1);
    const top = ranked.slice(0, cutPoint);
    const remainder = ranked.slice(cutPoint);
    if (remainder.length)
      top.push(NetworkVisualization.#mergeOverflowGroups(remainder));
    // Order by qualitative activation label bucket: positive high -> negative high
    const ordered = NetworkVisualization.#safeToSorted(
      top,
      NetworkVisualization.#activationLabelComparator
    );
    return {
      finalGroups: ordered.map((m) => m.group),
      finalLabels: ordered.map((m) => m.label),
    };
  }

  /** Merge overflow group metadata into a single synthetic bucket. */
  static #mergeOverflowGroups(
    metadataList: { group: any[]; label: string; size: number }[]
  ) {
    // Use reduce with spread push; could also use flatMap but this is explicit.
    return metadataList.reduce(
      (acc, current) => {
        acc.group.push(...current.group);
        acc.size += current.size;
        return acc;
      },
      { group: [] as any[], label: 'other±', size: 0 }
    );
  }

  /** Safe wrapper around ES2023 Array.prototype.toSorted with graceful fallback. */
  static #safeToSorted<T>(array: T[], compare: (a: T, b: T) => number): T[] {
    const anyArray: any = array as any;
    if (typeof anyArray.toSorted === 'function')
      return anyArray.toSorted(compare);
    return [...array].sort(compare);
  }

  /** Comparator for activation range label ordering (heuristic). */
  static #activationLabelComparator(a: any, b: any): number {
    const aNeg = a.label.includes('-');
    const bNeg = b.label.includes('-');
    if (aNeg !== bNeg) return aNeg ? 1 : -1; // positives first
    // Very high variants first within polarity
    const veryA = a.label.startsWith('v-high');
    const veryB = b.label.startsWith('v-high');
    if (veryA !== veryB) return veryA ? -1 : 1;
    const highA = a.label.includes('high');
    const highB = b.label.includes('high');
    if (highA !== highB) return highA ? -1 : 1;
    return 0;
  }

  /**
   * Utility to create a visualization node from a neataptic node.
   *
   * @param node - Neural network node object.
   * @param index - Index of the node in the network.
   * @returns Visualization node object.
   */
  static #toVisualizationNode(node: any, index: number): IVisualizationNode {
    // Use node.index if available, else fallback to array index
    const id = typeof node.index === 'number' ? node.index : index;
    return {
      id,
      uuid: String(id),
      type: node.type,
      activation: node.activation,
      bias: node.bias,
    };
  }

  /** Categorize nodes in a single pass (avoids 3 separate filter passes). */
  static #categorizeNodes(
    network: INetwork
  ): {
    inputNodes: IVisualizationNode[];
    hiddenNodes: IVisualizationNode[];
    outputNodes: IVisualizationNode[];
    inputCountDetected: number;
  } {
    const inputNodes: IVisualizationNode[] = [];
    const hiddenNodes: IVisualizationNode[] = [];
    const outputNodes: IVisualizationNode[] = [];
    const nodes = network.nodes || [];
    for (let index = 0; index < nodes.length; index++) {
      const node = nodes[index];
      const viz = NetworkVisualization.#toVisualizationNode(node, index);
      switch (node.type) {
        case 'input':
        case 'constant':
          inputNodes.push(viz);
          break;
        case 'hidden':
          hiddenNodes.push(viz);
          break;
        case 'output':
          outputNodes.push(viz);
          break;
        default:
          // ignore other experimental types
          break;
      }
    }
    return {
      inputNodes,
      hiddenNodes,
      outputNodes,
      inputCountDetected: inputNodes.length,
    };
  }

  /** Ensure connection-count scratch buffer is large enough. */
  static #ensureConnectionScratch(required: number): Int32Array {
    if (NetworkVisualization.#ConnectionCountsScratch.length < required) {
      let newSize = NetworkVisualization.#ConnectionCountsScratch.length;
      while (newSize < required) newSize *= 2;
      NetworkVisualization.#ConnectionCountsScratch = new Int32Array(newSize);
    }
    return NetworkVisualization.#ConnectionCountsScratch;
  }

  /** Compute connection counts between sequential layer boundaries. */
  static #computeConnectionCounts(
    network: INetwork,
    inputNodes: IVisualizationNode[],
    hiddenLayers: any[][],
    outputNodes: IVisualizationNode[]
  ): Int32Array {
    const numHiddenLayers = hiddenLayers.length;
    const arrows = numHiddenLayers > 0 ? numHiddenLayers + 1 : 1; // input->first/ output plus between hidden
    const scratch = NetworkVisualization.#ensureConnectionScratch(arrows);
    scratch.fill(0, 0, arrows);
    NetworkVisualization.#ConnectionCountsLen = arrows;
    const inputIdSet = new Set<number>(inputNodes.map((n) => Number(n.id)));
    const outputIdSet = new Set<number>(outputNodes.map((n) => Number(n.id)));
    const hiddenSets: Set<number>[] = hiddenLayers.map(
      (layer) => new Set(layer.map((n: any) => Number(n.id)))
    );
    const connections = network.connections || [];
    for (let ci = 0; ci < connections.length; ci++) {
      const conn: any = connections[ci];
      const fromIdx = Number(conn.from?.index ?? -1);
      const toIdx = Number(conn.to?.index ?? -1);
      // input -> first target
      if (inputIdSet.has(fromIdx)) {
        if (hiddenSets[0] && hiddenSets[0].has(toIdx)) scratch[0]++;
        else if (hiddenSets.length === 0 && outputIdSet.has(toIdx))
          scratch[0]++;
        continue;
      }
      // hidden -> hidden or hidden -> output
      for (let h = 0; h < hiddenSets.length; h++) {
        const fromSet = hiddenSets[h];
        if (!fromSet.has(fromIdx)) continue;
        const lastHidden = h === hiddenSets.length - 1;
        if (!lastHidden) {
          const toSet = hiddenSets[h + 1];
          if (toSet.has(toIdx)) {
            scratch[1 + h]++;
            break;
          }
        } else {
          if (outputIdSet.has(toIdx)) {
            scratch[hiddenSets.length]++;
            break;
          }
        }
      }
    }
    return scratch;
  }

  /** Build header string. */
  static #buildHeader(
    inputCount: number,
    hiddenLayers: any[][],
    outputCount: number,
    connectionCounts: Int32Array
  ): string {
    const numHiddenLayers = hiddenLayers.length;
    const { columnWidth } = NetworkVisualization.#computeLayout(
      numHiddenLayers
    );
    const parts = NetworkVisualization.#ScratchHeaderParts;
    parts.length = 0;
    parts.push(
      NetworkVisualization.#buildHeaderSegment({
        prefix: `${colors.blueCore}║`,
        label: `${colors.neonGreen}Input Layer [${inputCount}]${colors.reset}`,
        width: columnWidth - 1,
      })
    );
    parts.push(
      NetworkVisualization.#formatHeaderArrow(connectionCounts[0] ?? 0)
    );
    for (
      let hiddenLayerIndex = 0;
      hiddenLayerIndex < numHiddenLayers;
      hiddenLayerIndex++
    ) {
      parts.push(
        NetworkVisualization.pad(
          `${colors.cyanNeon}Hidden ${hiddenLayerIndex + 1} [${
            hiddenLayers[hiddenLayerIndex].length
          }]${colors.reset}`,
          columnWidth
        )
      );
      parts.push(
        NetworkVisualization.#formatHeaderArrow(
          connectionCounts[hiddenLayerIndex + 1] || 0
        )
      );
    }
    parts.push(
      NetworkVisualization.pad(
        `${colors.orangeNeon}Output Layer [${outputCount}]${colors.reset}`,
        columnWidth,
        ' ',
        'center'
      ) + `${colors.blueCore}║${colors.reset}`
    );
    return parts.join('');
  }

  /** Build a single header segment with optional prefix. */
  static #buildHeaderSegment(params: {
    prefix?: string;
    label: string;
    width: number;
  }): string {
    const { prefix = '', label, width } = params;
    return prefix + NetworkVisualization.pad(label, width, ' ', 'center');
  }

  /** Format a header arrow segment including connection count label. */
  static #formatHeaderArrow(connectionCount: number): string {
    const text = `${
      colors.blueNeon
    }${connectionCount} ${NetworkVisualization.#ARROW.trim()}${colors.reset}`;
    return NetworkVisualization.pad(text, NetworkVisualization.#ARROW_WIDTH);
  }

  /** Build legend footer lines. */
  static #buildLegend(): string[] {
    return [
      // Spacer
      `${colors.blueCore}║       ${NetworkVisualization.pad(' ', 140)} ║${
        colors.reset
      }`,
      `${colors.blueCore}║       ${NetworkVisualization.pad(
        'Arrows indicate feed-forward flow.',
        140,
        ' ',
        'left'
      )} ${colors.blueCore}║${colors.reset}`,
      `${colors.blueCore}║       ${NetworkVisualization.pad(' ', 140)} ║${
        colors.reset
      }`,
      `${colors.blueCore}║       ${NetworkVisualization.pad(
        `${colors.whiteNeon}Legend:  ${colors.neonGreen}●${colors.reset}=Input                    ${colors.cyanNeon}■${colors.reset}=Hidden                    ${colors.orangeNeon}▲${colors.reset}=Output`,
        140,
        ' ',
        'left'
      )} ${colors.blueCore}║${colors.reset}`,
      `${colors.blueCore}║       ${NetworkVisualization.pad(
        `${colors.whiteNeon}Groups:  ${colors.bgOrangeNeon}${colors.bright}v-high+${colors.reset}=Very high positive   ${colors.orangeNeon}high+${colors.reset}=High positive    ${colors.cyanNeon}mid+${colors.reset}=Medium positive    ${colors.neonGreen}low+${colors.reset}=Low positive`,
        140,
        ' ',
        'left'
      )} ${colors.blueCore}║${colors.reset}`,
      `${colors.blueCore}║       ${NetworkVisualization.pad(
        `${colors.whiteNeon}         zero±${colors.reset}=Near zero`,
        140,
        ' ',
        'left'
      )} ${colors.blueCore}║${colors.reset}`,
      `${colors.blueCore}║       ${NetworkVisualization.pad(
        `         ${colors.bgBlueCore}${colors.bright}v-high-${colors.reset}=Very high negative   ${colors.blueNeon}${colors.bright}high-${colors.reset}=High negative    ${colors.blueCore}mid-${colors.reset}=Medium negative    ${colors.blue}low-${colors.reset}=Low negative`,
        140,
        ' ',
        'left'
      )} ${colors.blueCore}║${colors.reset}`,
    ];
  }

  /** Build row strings for body. */
  static #buildRows(
    params: {
      inputCount: number;
      outputCount: number;
      inputNodes: IVisualizationNode[];
      displayLayers: any[][];
      layerDisplayCounts: number[];
      outputNodes: IVisualizationNode[];
      connectionCounts: Int32Array;
    },
    columnWidth: number
  ): string[] {
    const context = NetworkVisualization.#buildRowsInit(params);
    const {
      maxRows,
      rows,
      inputDisplayNodes,
      outputDisplayNodes,
      numHiddenLayers,
      inputCount,
      outputCount,
      inputNodes,
      displayLayers,
      connectionCounts,
    } = context;
    for (let rowIndex = 0; rowIndex < maxRows; rowIndex++) {
      let line = '';
      line += NetworkVisualization.#buildInputCell({
        rowIndex,
        inputCount,
        columnWidth,
        inputDisplayNodes,
      });
      line += NetworkVisualization.#buildFirstArrowCell({
        rowIndex,
        inputCount,
        inputNodes,
        displayLayers,
        connectionCounts,
      });
      for (let layerIndex = 0; layerIndex < numHiddenLayers; layerIndex++) {
        line += NetworkVisualization.#buildHiddenLayerCell({
          rowIndex,
          layerIndex,
          columnWidth,
          displayLayers,
        });
        line += NetworkVisualization.#buildInterLayerArrowCell({
          rowIndex,
          layerIndex,
          numHiddenLayers,
          displayLayers,
          connectionCounts,
          outputCount,
        });
      }
      line += NetworkVisualization.#buildOutputCell({
        rowIndex,
        outputCount,
        outputDisplayNodes,
        columnWidth,
      });
      rows.push(line);
    }
    return rows.slice();
  }

  /** Initialize reusable structures for row building. */
  static #buildRowsInit(params: {
    inputCount: number;
    outputCount: number;
    inputNodes: IVisualizationNode[];
    displayLayers: any[][];
    layerDisplayCounts: number[];
    outputNodes: IVisualizationNode[];
    connectionCounts: Int32Array;
  }) {
    const {
      inputCount,
      outputCount,
      inputNodes,
      displayLayers,
      layerDisplayCounts,
      outputNodes,
      connectionCounts,
    } = params;
    const maxRows = Math.max(inputCount, ...layerDisplayCounts, outputCount);
    const rows = NetworkVisualization.#ScratchRows;
    rows.length = 0;
    const inputDisplayNodes = Array.from(
      { length: inputCount },
      (_, idx) => inputNodes[idx] || { activation: 0 }
    );
    const outputDisplayNodes = Array.from(
      { length: outputCount },
      (_, idx) => outputNodes[idx] || { activation: 0 }
    );
    return {
      maxRows,
      rows,
      inputDisplayNodes,
      outputDisplayNodes,
      numHiddenLayers: displayLayers.length,
      inputCount,
      outputCount,
      inputNodes,
      displayLayers,
      connectionCounts,
    };
  }

  /** Build cell for an input row (including label). */
  static #buildInputCell(params: {
    rowIndex: number;
    inputCount: number;
    columnWidth: number;
    inputDisplayNodes: any[];
  }): string {
    const { rowIndex, inputCount, columnWidth, inputDisplayNodes } = params;
    if (rowIndex >= inputCount)
      return NetworkVisualization.pad('', columnWidth);
    const INPUT_LABELS6 = [
      'compass',
      'openN',
      'openE',
      'openS',
      'openW',
      'progress',
    ];
    const node = inputDisplayNodes[rowIndex];
    const label = rowIndex < 6 ? INPUT_LABELS6[rowIndex] : '';
    const labelStr = label ? ` ${colors.whiteNeon}${label}${colors.reset}` : '';
    return NetworkVisualization.pad(
      `${colors.blueCore}║   ${NetworkVisualization.#formatNode(
        colors.neonGreen,
        '●',
        node,
        labelStr
      )}`,
      columnWidth,
      ' ',
      'left'
    );
  }

  /** Build arrow cell between input and first hidden layer. */
  static #buildFirstArrowCell(params: {
    rowIndex: number;
    inputCount: number;
    inputNodes: IVisualizationNode[];
    displayLayers: any[][];
    connectionCounts: Int32Array;
  }): string {
    const {
      rowIndex,
      inputCount,
      inputNodes,
      displayLayers,
      connectionCounts,
    } = params;
    const firstHiddenTotal = displayLayers[0]?.length || 0;
    const totalInputs = Math.min(inputCount, inputNodes.length);
    const base = `${colors.blueNeon}${NetworkVisualization.#ARROW}${
      colors.reset
    }`;
    if (rowIndex === 0 && totalInputs && firstHiddenTotal) {
      const nodeProportion = Math.ceil(
        (connectionCounts[0] || 0) / Math.max(1, totalInputs)
      );
      return NetworkVisualization.pad(
        `${colors.blueNeon}${nodeProportion} ──▶${colors.reset}`,
        NetworkVisualization.#ARROW_WIDTH
      );
    }
    if (
      rowIndex < inputCount &&
      rowIndex < firstHiddenTotal &&
      totalInputs &&
      firstHiddenTotal
    ) {
      const nodeProportion = Math.ceil(
        (connectionCounts[0] || 0) / Math.max(3, totalInputs * 2)
      );
      return NetworkVisualization.pad(
        `${colors.blueNeon}${nodeProportion} ──▶${colors.reset}`,
        NetworkVisualization.#ARROW_WIDTH
      );
    }
    return NetworkVisualization.pad(base, NetworkVisualization.#ARROW_WIDTH);
  }

  /** Build hidden layer node cell. */
  static #buildHiddenLayerCell(params: {
    rowIndex: number;
    layerIndex: number;
    columnWidth: number;
    displayLayers: any[][];
  }): string {
    const { rowIndex, layerIndex, columnWidth, displayLayers } = params;
    const layer = displayLayers[layerIndex];
    if (rowIndex >= layer.length)
      return NetworkVisualization.pad(' ', columnWidth);
    const node = layer[rowIndex];
    if (node.isAverage) {
      const labelText = node.label ? `${node.label} ` : '';
      const extra = ` ${colors.dim}(${labelText}avg of ${node.avgCount})${colors.reset}`;
      return NetworkVisualization.pad(
        NetworkVisualization.#formatNode(colors.cyanNeon, '■', node, extra),
        columnWidth,
        ' ',
        'left'
      );
    }
    return NetworkVisualization.pad(
      NetworkVisualization.#formatNode(colors.cyanNeon, '■', node),
      columnWidth,
      ' ',
      'left'
    );
  }

  /** Build arrow cell either between hidden layers or from last hidden to outputs. */
  static #buildInterLayerArrowCell(params: {
    rowIndex: number;
    layerIndex: number;
    numHiddenLayers: number;
    displayLayers: any[][];
    connectionCounts: Int32Array;
    outputCount: number;
  }): string {
    const {
      rowIndex,
      layerIndex,
      numHiddenLayers,
      displayLayers,
      connectionCounts,
      outputCount,
    } = params;
    const layer = displayLayers[layerIndex];
    const arrowPlaceholder = `${colors.blueNeon}${NetworkVisualization.#ARROW}${
      colors.reset
    }`;
    const isLast = layerIndex === numHiddenLayers - 1;
    if (!isLast) {
      const connCount = connectionCounts[layerIndex + 1] || 0;
      if (rowIndex === 0) {
        const currentLayerSize = layer.length || 1;
        const nodeProportion = Math.ceil(
          connCount / Math.max(3, currentLayerSize * 2)
        );
        return NetworkVisualization.pad(
          `${colors.blueNeon}${nodeProportion} ──▶${colors.reset}`,
          NetworkVisualization.#ARROW_WIDTH
        );
      }
      if (
        rowIndex < layer.length &&
        rowIndex < (displayLayers[layerIndex + 1]?.length || 0)
      ) {
        const currentLayerSize = layer.length || 1;
        const proportion = Math.max(
          1,
          Math.min(5, Math.ceil(connCount / Math.max(3, currentLayerSize)))
        );
        return NetworkVisualization.pad(
          `${colors.blueNeon}${proportion} ──▶${colors.reset}`,
          NetworkVisualization.#ARROW_WIDTH
        );
      }
      return NetworkVisualization.pad(
        arrowPlaceholder,
        NetworkVisualization.#ARROW_WIDTH
      );
    }
    // Last hidden -> output
    const lastConnCount = connectionCounts[numHiddenLayers] || 0;
    if (rowIndex === 0) {
      const lastLayerSize = layer.length || 1;
      const nodeProportion = Math.ceil(
        lastConnCount / Math.max(3, lastLayerSize * 2)
      );
      return NetworkVisualization.pad(
        `${colors.blueNeon}${nodeProportion} ──▶${colors.reset}`,
        NetworkVisualization.#ARROW_WIDTH
      );
    }
    if (rowIndex < layer.length && rowIndex < outputCount) {
      const lastLayerSize = layer.length || 1;
      const proportion = Math.max(
        1,
        Math.min(5, Math.ceil(lastConnCount / Math.max(5, lastLayerSize * 2)))
      );
      return NetworkVisualization.pad(
        `${colors.blueNeon}${proportion} ──▶${colors.reset}`,
        NetworkVisualization.#ARROW_WIDTH
      );
    }
    return NetworkVisualization.pad(
      arrowPlaceholder,
      NetworkVisualization.#ARROW_WIDTH
    );
  }

  /** Build output layer cell. */
  static #buildOutputCell(params: {
    rowIndex: number;
    outputCount: number;
    outputDisplayNodes: any[];
    columnWidth: number;
  }): string {
    const { rowIndex, outputCount, outputDisplayNodes, columnWidth } = params;
    if (rowIndex >= outputCount)
      return NetworkVisualization.pad('', columnWidth);
    const node = outputDisplayNodes[rowIndex];
    return (
      NetworkVisualization.pad(
        NetworkVisualization.#formatNode(colors.orangeNeon, '▲', node),
        columnWidth,
        ' ',
        'left'
      ) + `${colors.blueCore}║${colors.reset}`
    );
  }

  /**
   * Generate a multi-line, colorized ASCII summary of the provided neural network.
   * The output includes:
   * - Layer headers with node counts and approximate connection counts between layers.
   * - Node activation values (numeric + color) for inputs, hidden (or averaged groups), and outputs.
   * - Condensed legend explaining symbols and activation grouping ranges.
   *
   * Hidden layer condensation: For large hidden layers, nodes are grouped into activation buckets;
   * each bucket is displayed as a single "average" virtual node whose value is the mean activation.
   * Buckets beyond the configured visible limit are merged into an "other±" meta-group.
   *
   * Performance: Uses internal scratch buffers to minimize intermediate allocations. Sorting relies
   * on ES2023 `toSorted` when available (with a stable fallback) ensuring deterministic grouping.
   *
   * @param network - The neural network (expects `nodes` and optional `connections`).
   * @returns Formatted multi-line string ready for terminal output (ANSI colors included).
   * @example
   * ```ts
   * import { NetworkVisualization } from './networkVisualization';
   * const ascii = NetworkVisualization.visualizeNetworkSummary(myNetwork);
   * console.log(ascii);
   * ```
   */
  static visualizeNetworkSummary(network: INetwork): string {
    const categorized = NetworkVisualization.#categorizeNodes(network);
    const INPUT_COUNT = categorized.inputCountDetected || 18; // legacy fallback retained
    const OUTPUT_COUNT = 4; // maze solver fixed
    const hiddenLayers = NetworkVisualization.#groupHiddenByLayer(
      categorized.inputNodes,
      categorized.hiddenNodes,
      categorized.outputNodes
    );
    const prepared = NetworkVisualization.#prepareHiddenLayersForDisplay(
      hiddenLayers
    );
    const connectionCounts = NetworkVisualization.#computeConnectionCounts(
      network,
      categorized.inputNodes,
      hiddenLayers,
      categorized.outputNodes
    );
    const { columnWidth } = NetworkVisualization.#computeLayout(
      hiddenLayers.length
    );
    const header = NetworkVisualization.#buildHeader(
      INPUT_COUNT,
      hiddenLayers,
      OUTPUT_COUNT,
      connectionCounts
    );
    const rows = NetworkVisualization.#buildRows(
      {
        inputCount: INPUT_COUNT,
        outputCount: OUTPUT_COUNT,
        inputNodes: categorized.inputNodes,
        displayLayers: prepared.displayLayers,
        layerDisplayCounts: prepared.layerDisplayCounts,
        outputNodes: categorized.outputNodes,
        connectionCounts,
      },
      columnWidth
    );
    const legendLines = NetworkVisualization.#buildLegend();
    return [header, ...rows, ...legendLines].join('\n');
  }

  /** Compute layout derived widths for given hidden layer count. */
  static #computeLayout(numHiddenLayers: number): { columnWidth: number } {
    const numLayers = 2 + numHiddenLayers;
    const numArrows = numLayers - 1;
    const availableWidth =
      NetworkVisualization.#TOTAL_WIDTH -
      numArrows * NetworkVisualization.#ARROW_WIDTH;
    const columnWidth = Math.floor(availableWidth / numLayers);
    return { columnWidth };
  }
}
