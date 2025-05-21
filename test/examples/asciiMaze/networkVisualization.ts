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
import { colors } from './colors';
import { IVisualizationNode, IVisualizationConnection } from './interfaces';

/**
 * NetworkVisualization provides static methods for visualizing neural networks in the terminal.
 * It includes utilities for formatting, grouping, and rendering network structure and activations.
 */
export class NetworkVisualization {
  /**
   * Pads a string to a specific width with alignment options.
   *
   * @param str - String to pad.
   * @param width - Target width for the string.
   * @param padChar - Character to use for padding (default: space).
   * @param align - Alignment option ('left', 'center', or 'right').
   * @returns Padded string of specified width with chosen alignment.
   */
  static pad(str: string, width: number, padChar: string = ' ', align: 'left'|'center'|'right' = 'center'): string {
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
  static getNodeValue(node: any): number {
    if (typeof node.activation === 'number' && isFinite(node.activation) && !isNaN(node.activation)) {
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
  static getActivationColor(value: number): string {
    // Use TRON-inspired color palette for activation values
    if (value >= 2.0) return colors.bgOrangeNeon + colors.bright;  // Very high positive
    if (value >= 1.0) return colors.orangeNeon;                   // High positive
    if (value >= 0.5) return colors.cyanNeon;                     // Medium positive
    if (value >= 0.1) return colors.neonGreen;                        // Low positive
    if (value >= -0.1) return colors.whiteNeon;                   // Near zero
    if (value >= -0.5) return colors.blue;                        // Low negative
    if (value >= -1.0) return colors.blueCore;                    // Medium negative
    if (value >= -2.0) return colors.bgNeonAqua + colors.bright;    // High negative
    return colors.bgNeonViolet + colors.neonSilver;                     // Very high negative
  }

  /**
   * Formats a numeric value for display with color based on its value.
   *
   * @param v - Numeric value to format.
   * @returns Colorized string representation of the value.
   */
  static fmtColoredValue(v: number): string {
    if (typeof v !== 'number' || isNaN(v) || !isFinite(v)) return ' 0.000';
    
    const color = NetworkVisualization.getActivationColor(v);
    let formattedValue;
    
    formattedValue = (v >= 0 ? ' ' : '') + v.toFixed(6);
    
    return color + formattedValue + colors.reset;
  }

  /**
   * Groups hidden nodes into layers based on their connections.
   *
   * @param inputNodes - Array of input nodes.
   * @param hiddenNodes - Array of hidden nodes.
   * @param outputNodes - Array of output nodes.
   * @returns Array of hidden node arrays, each representing a layer.
   */
  static groupHiddenByLayer(inputNodes: any[], hiddenNodes: any[], outputNodes: any[]): any[][] {
    if (hiddenNodes.length === 0) return [];
    
    let layers: any[][] = [];
    let prevLayer = inputNodes;
    let remaining = [...hiddenNodes];
    
    while (remaining.length > 0) {
      const currentLayer = remaining.filter(h =>
        h.connections && h.connections.in && h.connections.in.length > 0 &&
        h.connections.in.every((conn: any) => prevLayer.includes(conn.from))
      );
      
      if (currentLayer.length === 0) {
        layers.push(remaining);
        break;
      }
      
      layers.push(currentLayer);
      prevLayer = currentLayer;
      remaining = remaining.filter(h => !currentLayer.includes(h));
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
  static groupNodesByActivation(nodes: any[]): {
    groups: any[][],
    labels: string[]
  } {
    // Calculate activation values
    const activations = nodes.map(node => NetworkVisualization.getNodeValue(node));
    
    // Define more granular activation ranges
    const ranges = [
      { min: 2.0, max: Infinity, label: "v-high+" },
      { min: 1.0, max: 2.0, label: "high+" },
      { min: 0.5, max: 1.0, label: "mid+" },
      { min: 0.1, max: 0.5, label: "low+" },
      { min: -0.1, max: 0.1, label: "zero±" },
      { min: -0.5, max: -0.1, label: "low-" },
      { min: -1.0, max: -0.5, label: "mid-" },
      { min: -2.0, max: -1.0, label: "high-" },
      { min: -Infinity, max: -2.0, label: "v-high-" }
    ];
    
    // Create groups and corresponding labels
    const groups: any[][] = [];
    const labels: string[] = [];
    
    // Group nodes by activation ranges
    for (const range of ranges) {
      const nodesInRange = nodes.filter((_, i) => 
        activations[i] >= range.min && activations[i] < range.max);
      
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
  static prepareHiddenLayersForDisplay(hiddenLayers: any[][], maxVisiblePerLayer: number = 10): {
    displayLayers: any[][],
    layerDisplayCounts: number[],
    averageNodes: {[key: string]: {avgValue: number, count: number}}
  } {
    const MAX_VISIBLE = maxVisiblePerLayer;
    
    const averageNodes: {[key: string]: {avgValue: number, count: number}} = {};
    const displayLayers: any[][] = [];
    const layerDisplayCounts: number[] = [];
    
    hiddenLayers.forEach((layer, layerIdx) => {
      if (layer.length <= MAX_VISIBLE) {
        // If layer is small enough, show all nodes
        displayLayers.push([...layer]);
        layerDisplayCounts.push(layer.length);
      } else {
        // For large layers, show all nodes as averages to better represent distribution
        
        // Group nodes by activation values
        const { groups, labels } = NetworkVisualization.groupNodesByActivation(layer);
        
        // If we have too many groups, we need to merge some to fit in MAX_VISIBLE
        let finalGroups = groups;
        let finalLabels = labels;
        
        if (groups.length > MAX_VISIBLE) {
          // We'll prioritize groups with more nodes and merge smaller ones
          const rankedGroups = groups.map((g, i) => ({
            group: g,
            label: labels[i],
            size: g.length
          })).sort((a, b) => b.size - a.size);
          
          // Take top MAX_VISIBLE-1 groups
          const topGroups = rankedGroups.slice(0, MAX_VISIBLE - 1);
          
          // Combine all remaining small groups
          const remainingGroups = rankedGroups.slice(MAX_VISIBLE - 1);
          const mergedGroup = remainingGroups.reduce((acc, curr) => {
            acc.group = [...acc.group, ...curr.group];
            return acc;
          }, { group: [], label: "other±", size: 0 });
          
          if (mergedGroup.group.length > 0) {
            topGroups.push(mergedGroup);
          }
          
          // Sort back to original order by activation range (assumed by label)
          topGroups.sort((a, b) => {
            // Sort by activation range - very high+ first, very high- last
            const aIsNegative = a.label.includes('-');
            const bIsNegative = b.label.includes('-');
            
            if (aIsNegative && !bIsNegative) return 1;
            if (!aIsNegative && bIsNegative) return -1;
            
            if (a.label.includes('v-') && !b.label.includes('v-')) return aIsNegative ? 1 : -1;
            if (!a.label.includes('v-') && b.label.includes('v-')) return aIsNegative ? -1 : 1;
            
            if (a.label.includes('high') && !b.label.includes('high')) return aIsNegative ? 1 : -1;
            if (!a.label.includes('high') && b.label.includes('high')) return aIsNegative ? -1 : 1;
            
            return 0;
          });
          
          finalGroups = topGroups.map(g => g.group);
          finalLabels = topGroups.map(g => g.label);
        }
        
        // Create "virtual" average nodes for each group
        const avgNodes = finalGroups.map((group, groupIdx) => {
          const avgKey = `layer${layerIdx}-avg-${groupIdx}`;
          const sum = group.reduce((acc: number, node: any) => acc + NetworkVisualization.getNodeValue(node), 0);
          const avgValue = group.length > 0 ? sum / group.length : 0;
          
          // Store average node info
          averageNodes[avgKey] = {
            avgValue,
            count: group.length
          };
          
          // Create a "virtual" average node to display
          return {
            id: -1 * (layerIdx * 1000 + groupIdx),
            uuid: avgKey,
            type: 'hidden',
            activation: avgValue,
            isAverage: true,
            avgCount: group.length,
            label: finalLabels[groupIdx]
          };
        });
        
        // Add only the average nodes
        displayLayers.push(avgNodes);
        layerDisplayCounts.push(avgNodes.length);
      }
    });
    
    return { displayLayers, layerDisplayCounts, averageNodes };
  }

  /**
   * Utility to create a visualization node from a neataptic node.
   *
   * @param node - Neural network node object.
   * @param index - Index of the node in the network.
   * @returns Visualization node object.
   */
  static toVisualizationNode(node: any, index: number): IVisualizationNode {
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

  /**
   * Visualizes a neural network's structure and activations in ASCII format.
   *
   * Creates a comprehensive terminal-friendly visualization showing:
   * - Network architecture with layers
   * - Node activation values with color coding
   * - Connection counts between layers
   * - Condensed representation of large hidden layers
   *
   * @param network - The neural network to visualize.
   * @returns String containing the ASCII visualization.
   */
  static visualizeNetworkSummary(network: INetwork): string {
    // Constants for visualization
    const ARROW = '  ──▶  ';
    const ARROW_WIDTH = ARROW.length;
    const TOTAL_WIDTH = 150; // Width of the entire visualization
    const INPUT_COUNT = 5;  // Number of input nodes (hardcoded for maze solver)
    const OUTPUT_COUNT = 4; // Number of output nodes (hardcoded for maze solver)

    // Extract nodes from network
    const nodes = network.nodes || [];
    const inputNodes: IVisualizationNode[] = nodes.filter(n => n.type === 'input' || n.type === 'constant').map(NetworkVisualization.toVisualizationNode);
    const outputNodes: IVisualizationNode[] = nodes.filter(n => n.type === 'output').map(NetworkVisualization.toVisualizationNode);
    const hiddenNodesRaw: IVisualizationNode[] = nodes.filter(n => n.type === 'hidden').map(NetworkVisualization.toVisualizationNode);

    // Group hidden nodes into layers
    const hiddenLayers = NetworkVisualization.groupHiddenByLayer(inputNodes, hiddenNodesRaw, outputNodes);
    const numHiddenLayers = hiddenLayers.length;
    
    // Prepare hidden layers for display (condensing large layers)
    const { displayLayers, layerDisplayCounts, averageNodes } = NetworkVisualization.prepareHiddenLayersForDisplay(hiddenLayers);

    // Map connections using node index as unique identifier
    const connections: IVisualizationConnection[] = (network.connections || []).map((conn: any) => ({
      weight: conn.weight,
      fromUUID: String(conn.from.index), // Use .index directly as per INodeStruct
      toUUID: String(conn.to.index),     // Use .index directly as per INodeStruct
      gaterUUID: conn.gater ? String(conn.gater.index) : null, // Use .index directly
      enabled: typeof conn.enabled === 'boolean' ? conn.enabled : true
    }));

    // Calculate connection counts between layers
    const connectionCounts: number[] = [];
    
    // Count input → first hidden (or output if no hidden)
    let firstCount = 0;
    const firstTargetLayer = hiddenLayers.length > 0 ? hiddenLayers[0] : outputNodes;
    for (const conn of (network.connections || [])) { // Add null check for network.connections
      if (inputNodes.some(n => n.id === conn.from.index) && firstTargetLayer.some(n => n.id === conn.to.index)) {
        firstCount++;
      }
    }
    connectionCounts.push(firstCount);
    
    // Count between hidden layers
    for (let i = 0; i < hiddenLayers.length - 1; i++) {
      let count = 0;
      for (const conn of (network.connections || [])) { // Add null check
        if (hiddenLayers[i].some(n => n.id === conn.from.index) && hiddenLayers[i+1].some(n => n.id === conn.to.index)) {
          count++;
        }
      }
      connectionCounts.push(count);
    }
    
    // Count last hidden → output
    if (hiddenLayers.length > 0) {
      let lastCount = 0;
      for (const conn of (network.connections || [])) { // Add null check
        if (hiddenLayers[hiddenLayers.length - 1].some(n => n.id === conn.from.index) && outputNodes.some(n => n.id === conn.to.index)) {
          lastCount++;
        }
      }
      connectionCounts.push(lastCount);
    }

    // --- Layer/connection summary footer ---

    // Calculate layout
    const numLayers = 2 + numHiddenLayers; // input + hidden + output
    const numArrows = numLayers - 1;
    const availableWidth = TOTAL_WIDTH - (numArrows * ARROW_WIDTH);
    const columnWidth = Math.floor(availableWidth / numLayers);
    
    // Create the header row
    let header = '';
    header += `${colors.blueCore}║` + NetworkVisualization.pad(`${colors.neonGreen}Input Layer [${INPUT_COUNT}]${colors.reset}`, columnWidth - 1);
    
    // First arrow with connection count on the left
    const firstConnCount = connectionCounts[0];
    const firstArrowText = `${colors.blueNeon}${firstConnCount} ${ARROW.trim()}${colors.reset}`;
    header += NetworkVisualization.pad(firstArrowText, ARROW_WIDTH);
    
    // Add hidden layer headers with connection counts
    for (let i = 0; i < numHiddenLayers; i++) {
      header += NetworkVisualization.pad(`${colors.cyanNeon}Hidden ${i+1} [${hiddenLayers[i].length}]${colors.reset}`, columnWidth);
      
      if (i < numHiddenLayers) {
        // Arrow with connection count on the left
        const connCount = connectionCounts[i+1] || 0;
        const arrowText = `${colors.blueNeon}${connCount} ${ARROW.trim()}${colors.reset}`;
        header += NetworkVisualization.pad(arrowText, ARROW_WIDTH);
      }
    }
    
    header += NetworkVisualization.pad(`${colors.orangeNeon}Output Layer [${OUTPUT_COUNT}]${colors.reset}`, columnWidth, ' ', 'center') + `${colors.blueCore}║${colors.reset}`;

    // Prepare display data for each layer
    // For input nodes: Always show all 9
    const inputDisplayNodes = Array(INPUT_COUNT).fill(null).map((_, i) => 
      inputNodes[i] || { activation: 0 }
    );
    
    // For output nodes: Always show all 4
    const outputDisplayNodes = Array(OUTPUT_COUNT).fill(null).map((_, i) => 
      outputNodes[i] || { activation: 0 }
    );
    
    // Calculate the max number of rows needed
    const maxRows = Math.max(
      INPUT_COUNT,
      ...layerDisplayCounts,
      OUTPUT_COUNT
    );
    
    // Generate rows for visualization
    const rows: string[] = [];
    for (let rowIdx = 0; rowIdx < maxRows; rowIdx++) {
      let row = '';
      
      // Input column
      if (rowIdx < INPUT_COUNT) {
        const node = inputDisplayNodes[rowIdx];
        const value = NetworkVisualization.getNodeValue(node);
        row += NetworkVisualization.pad(`${colors.blueCore}║   ${colors.neonGreen}●${colors.reset}${NetworkVisualization.fmtColoredValue(value)}`, columnWidth, ' ', 'left');
      } else {
        row += NetworkVisualization.pad('', columnWidth);
      }
      
      // First arrow - calculate proportional connection counts
      if (rowIdx === 0) {
        // First row after header shows proportional counts, not total
        const totalInputs = Math.min(INPUT_COUNT, inputNodes.length);
        const firstHiddenTotal = displayLayers[0]?.length || 0;
        
        if (totalInputs > 0 && firstHiddenTotal > 0) {
          // Calculate a proportional number of connections for the first visible row
          const nodeProportion = Math.ceil(connectionCounts[0] / Math.max(1, totalInputs));
          row += NetworkVisualization.pad(`${colors.blueNeon}${nodeProportion} ──▶${colors.reset}`, ARROW_WIDTH);
        } else {
          row += NetworkVisualization.pad(`${colors.blueNeon}${ARROW}${colors.reset}`, ARROW_WIDTH);
        }
      } else if (rowIdx < INPUT_COUNT && rowIdx < displayLayers[0]?.length) {
        // Calculate proportional connections for this input node to first hidden layer
        const totalInputs = Math.min(INPUT_COUNT, inputNodes.length);
        const firstHiddenTotal = displayLayers[0]?.length || 0;
        
        if (totalInputs > 0 && firstHiddenTotal > 0) {
          // Calculate a proportional number of connections
          const nodeProportion = Math.ceil(connectionCounts[0] / Math.max(3, totalInputs * 2));
          row += NetworkVisualization.pad(`${colors.blueNeon}${nodeProportion} ──▶${colors.reset}`, ARROW_WIDTH);
        } else {
          row += NetworkVisualization.pad(`${colors.blueNeon}${ARROW}${colors.reset}`, ARROW_WIDTH);
        }
      } else {
        // For other rows, just show the arrow without a number
        row += NetworkVisualization.pad(`${colors.blueNeon}${ARROW}${colors.reset}`, ARROW_WIDTH);
      }
      
      // Hidden layers
      for (let layerIdx = 0; layerIdx < numHiddenLayers; layerIdx++) {
        const layer = displayLayers[layerIdx];
        if (rowIdx < layer.length) {
          const node = layer[rowIdx];
          
          if (node.isAverage) {
            // Special formatting for average nodes
            const labelText = node.label ? `${node.label} ` : '';
            const avgText = `${colors.cyanNeon}■${colors.reset}${NetworkVisualization.fmtColoredValue(node.activation)} ${colors.dim}(${labelText}avg of ${node.avgCount})${colors.reset}`;
            row += NetworkVisualization.pad(avgText, columnWidth, ' ', 'left');
          } else {
            const value = NetworkVisualization.getNodeValue(node);
            row += NetworkVisualization.pad(`${colors.cyanNeon}■${colors.reset}${NetworkVisualization.fmtColoredValue(value)}`, columnWidth, ' ', 'left');
          }
        } else {
          row += NetworkVisualization.pad(' ', columnWidth);
        }
        
        // Arrow between columns - calculate proportional connection counts
        if (layerIdx < numHiddenLayers - 1) {
          // Arrow to next hidden layer
          const connCount = connectionCounts[layerIdx + 1];
          if (rowIdx === 0) {
            // First row shows proportional connection count, not total
            const currentLayerSize = displayLayers[layerIdx]?.length || 1;
            const nodeProportion = Math.ceil(connCount / Math.max(3, currentLayerSize * 2));
            row += NetworkVisualization.pad(`${colors.blueNeon}${nodeProportion} ──▶${colors.reset}`, ARROW_WIDTH);
          } else if (rowIdx < layer.length && rowIdx < displayLayers[layerIdx + 1]?.length) {
            // Calculate proportional connections between these hidden layers
            const currentLayerSize = displayLayers[layerIdx]?.length || 1;
            const nextLayerSize = displayLayers[layerIdx + 1]?.length || 1;
            
            // For hidden → hidden connections, distribute more evenly based on layer sizes
            const proportion = Math.max(1, Math.min(5, Math.ceil(connCount / Math.max(3, currentLayerSize))));
            row += NetworkVisualization.pad(`${colors.blueNeon}${proportion} ──▶${colors.reset}`, ARROW_WIDTH);
          } else {
            // Otherwise just show arrow
            row += NetworkVisualization.pad(`${colors.blueNeon}${ARROW}${colors.reset}`, ARROW_WIDTH);
          }
        } else {
          // Last arrow to output layer
          const connCount = connectionCounts[connectionCounts.length - 1];
          if (rowIdx === 0) {
            // First row shows proportional connections, not total
            const lastLayerSize = displayLayers[displayLayers.length - 1]?.length || 1;
            const nodeProportion = Math.ceil(connCount / Math.max(3, lastLayerSize * 2));
            row += NetworkVisualization.pad(`${colors.blueNeon}${nodeProportion} ──▶${colors.reset}`, ARROW_WIDTH);
          } else if (rowIdx < layer.length && rowIdx < OUTPUT_COUNT) {
            // Calculate proportional connections to output layer
            const lastLayerSize = displayLayers[displayLayers.length - 1]?.length || 1;
            
            // For last hidden → output, calculate a reasonable proportion
            // This should show a small number, typically 1-5, not the total
            const proportion = Math.max(1, Math.min(5, Math.ceil(connCount / Math.max(5, lastLayerSize * 2))));
            row += NetworkVisualization.pad(`${colors.blueNeon}${proportion} ──▶${colors.reset}`, ARROW_WIDTH);
          } else {
            // Otherwise just show arrow
            row += NetworkVisualization.pad(`${colors.blueNeon}${ARROW}${colors.reset}`, ARROW_WIDTH);
          }
        }
      }
      
      // Output column - ALWAYS show all 4 outputs
      if (rowIdx < OUTPUT_COUNT) {
        const node = outputDisplayNodes[rowIdx];
        const value = NetworkVisualization.getNodeValue(node);
        row += NetworkVisualization.pad(`${colors.orangeNeon}▲${colors.reset}${NetworkVisualization.fmtColoredValue(value)}`, columnWidth, ' ', 'left') + `${colors.blueCore}║${colors.reset}`;
      } else {
        row += NetworkVisualization.pad('', columnWidth);
      }
      
      rows.push(row);
    }
    
    // Combine all parts with a legend
    return [
      header,
      ...rows,
      `${colors.blueCore}║       ${NetworkVisualization.pad(' ', 140)} ║${colors.reset}`,
      `${colors.blueCore}║       ${NetworkVisualization.pad('Arrows indicate feed-forward flow.', 140, ' ', 'left')} ${colors.blueCore}║${colors.reset}`,
      `${colors.blueCore}║       ${NetworkVisualization.pad(' ', 140)} ║${colors.reset}`,

      `${colors.blueCore}║       ${NetworkVisualization.pad(`${colors.whiteNeon}Legend:  ${colors.neonGreen}●${colors.reset}=Input                    ${colors.cyanNeon}■${colors.reset}=Hidden                    ${colors.orangeNeon}▲${colors.reset}=Output`, 140, ' ', 'left')} ${colors.blueCore}║${colors.reset}`,
      `${colors.blueCore}║       ${NetworkVisualization.pad(`${colors.whiteNeon}Groups:  ${colors.bgOrangeNeon}${colors.bright}v-high+${colors.reset}=Very high positive   ${colors.orangeNeon}high+${colors.reset}=High positive    ${colors.cyanNeon}mid+${colors.reset}=Medium positive    ${colors.neonGreen}low+${colors.reset}=Low positive`, 140, ' ', 'left')} ${colors.blueCore}║${colors.reset}`,
      `${colors.blueCore}║       ${NetworkVisualization.pad(`${colors.whiteNeon}         zero±${colors.reset}=Near zero`, 140, ' ', 'left')} ${colors.blueCore}║${colors.reset}`,
      `${colors.blueCore}║       ${NetworkVisualization.pad(`         ${colors.bgBlueCore}${colors.bright}v-high-${colors.reset}=Very high negative   ${colors.blueNeon}${colors.bright}high-${colors.reset}=High negative    ${colors.blueCore}mid-${colors.reset}=Medium negative    ${colors.blue}low-${colors.reset}=Low negative`, 140, ' ', 'left')} ${colors.blueCore}║${colors.reset}`,
    ].join('\n');
  }
}