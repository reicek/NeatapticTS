/**
 * Red-phase test contracts for slice A3: NGE main-agent embryo materialization.
 *
 * The bridge `src/neat/nge-main-agent/nge-to-network.ts` does not exist yet;
 * these tests define the observable contract for `materializeFromNgeState`.
 *
 * Follows the relaxed single-expect rule: up to three related assertions are
 * allowed when they verify the same behavior state; independent contracts are
 * split into separate `it()` blocks.
 */

import Connection from '../../architecture/connection';
import Network from '../../architecture/network/network';

import { buildMainAgentEmbryo } from './neat.nge-main-agent.embryo';
import { materializeFromNgeState } from './nge-to-network';
import type { NgeMainAgentLifecycleConfig } from './neat.nge-main-agent.types';

const lifecycleConfig: NgeMainAgentLifecycleConfig = {
  seed: 42,
  maxNodes: 1024,
  maxEdges: 4096,
};

function buildEmbryo() {
  return buildMainAgentEmbryo(lifecycleConfig);
}

describe('materializeFromNgeState', () => {
  it('exports a function that returns a Network instance from an embryo', () => {
    // Arrange
    const embryo = buildEmbryo();

    // Act
    const network = materializeFromNgeState(embryo);

    // Assert
    expect(typeof materializeFromNgeState).toBe('function');
    expect(network).toBeInstanceOf(Network);
  });

  it('reflects embryo topology counts in network nodes and connections', () => {
    // Arrange
    const embryo = buildEmbryo();

    // Act
    const network = materializeFromNgeState(embryo);

    // Assert
    expect(network.nodes.length).toBe(embryo.nodeCount);
    expect(network.connections.length + network.selfconns.length).toBe(
      embryo.edgeCount,
    );
  });

  it('produces identical topology counts for the same embryo and seed across calls', () => {
    // Arrange
    const embryo = buildEmbryo();

    // Act
    const networkA = materializeFromNgeState(embryo);
    const networkB = materializeFromNgeState(embryo);

    // Assert
    expect(networkA.nodes.length).toBe(networkB.nodes.length);
    expect(networkA.connections.length + networkA.selfconns.length).toBe(
      networkB.connections.length + networkB.selfconns.length,
    );
  });

  it('materializes at least one recurrent connection when embryo includes recurrent motifs', () => {
    // Arrange
    const embryo = buildEmbryo();

    // Act
    const network = materializeFromNgeState(embryo);

    // Assert — self-connections live in `selfconns` per the Network.construct
    // invariant, so check both arrays for a recurrent/self edge.
    const hasRecurrentConnection =
      network.connections.some(
        (connection: Connection) =>
          connection.from === connection.to ||
          (connection as { recurrent?: boolean }).recurrent === true,
      ) || network.selfconns.length > 0;
    expect(hasRecurrentConnection).toBe(true);
  });
});
