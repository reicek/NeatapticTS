/**
 * Shared type contracts for network-view overlays.
 *
 * The most notable overlay is the input-group label band system, which annotates
 * stacked temporal observation channels so the input layer reads as grouped
 * semantics instead of a flat strip of anonymous nodes.
 */

/**
 * Input-group label band geometry and style contract.
 *
 * Each band identifies a contiguous span of input nodes and the visual style
 * used to render that group marker.
 */
export interface InputGroupLabelBand {
  label: string;
  startNodeIndex: number;
  endNodeIndex: number;
  backgroundColor: string;
  orientation: 'vertical' | 'horizontal';
}
