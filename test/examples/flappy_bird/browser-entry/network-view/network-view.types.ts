/**
 * Input-group label band geometry and style contract.
 */
export interface InputGroupLabelBand {
  label: string;
  startNodeIndex: number;
  endNodeIndex: number;
  backgroundColor: string;
  orientation: 'vertical' | 'horizontal';
}
