/**
 * Shared type contracts for network-view overlays.
 *
 * The most notable overlays are the input-group label bands and the per-input
 * row descriptions. Together they turn the Flappy input shelf back into a
 * readable teaching surface instead of a flat strip of anonymous nodes.
 */

/**
 * Input-group label band geometry and style contract.
 *
 * Each band identifies a contiguous span of input nodes and the visual style
 * used to render that group marker.
 */
export interface InputGroupLabelBand {
  label: string;
  labelLines: readonly string[];
  tooltipHeading: string;
  tooltipBodyParagraphs: readonly string[];
  startNodeIndex: number;
  endNodeIndex: number;
  backgroundColor: string;
  orientation: 'vertical' | 'horizontal';
}

/**
 * One horizontal description aligned to a specific Flappy input node.
 *
 * The label sits between the semantic group band and the network itself so the
 * viewer can understand each observation channel without inspecting source.
 */
export interface InputNodeDescriptionLabel {
  labelLines: readonly string[];
  tooltipHeading: string;
  tooltipBodyParagraphs: readonly string[];
  nodeIndex: number;
}
