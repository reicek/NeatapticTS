import Network from '../../src/architecture/network';
import type { INetwork } from './interfaces';
import { NetworkVisualization } from './networkVisualization';

const ANSI_ESCAPE_REGEX = new RegExp(
  `${String.fromCharCode(27)}\\[[0-9;]*m`,
  'g',
);

describe('NetworkVisualization', () => {
  it('uses the runtime output-role count instead of the legacy fixed output count', () => {
    const network = new Network(6, 2, { seed: 612 });
    const summary = stripAnsi(
      NetworkVisualization.visualizeNetworkSummary(
        network as unknown as INetwork,
      ),
    );

    expect(summary).toContain('Output Layer [2]');
  });
});

function stripAnsi(value: string): string {
  return value.replace(ANSI_ESCAPE_REGEX, '');
}
