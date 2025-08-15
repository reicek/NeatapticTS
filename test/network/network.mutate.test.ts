import Network from '../../src/architecture/network';
import { mutateImpl } from '../../src/architecture/network/network.mutate';
import mutation from '../../src/methods/mutation';
import { config } from '../../src/config';

/**
 * Mutation dispatcher & operator edge cases.
 * Targets uncovered branches: unknown method no-op, deterministicChainMode ADD_NODE,
 * acyclic blocks for self/back connections, gating add/remove, activation bias guards, swap nodes output exclusion.
 */
describe('Network.mutateImpl & operators', () => {
  describe('Scenario: unknown mutation method', () => {
    it('silently no-ops (connections unchanged)', () => {
      // Arrange
      const net = new Network(2, 1, { seed: 1 });
      const before = net.connections.length;
      // Act
      mutateImpl.call(net, 'NON_EXISTENT_MUT');
      // Assert
      expect(net.connections.length).toBe(before);
    });
  });

  describe('Scenario: ADD_NODE deterministic chain mode', () => {
    it('extends linear chain depth by exactly one hidden node', () => {
      // Arrange
      (config as any).deterministicChainMode = true;
      const net = new Network(1, 1, { seed: 2 });
      const beforeHidden = net.nodes.filter((n) => n.type === 'hidden').length;
      // Act
      mutateImpl.call(net, mutation.ADD_NODE);
      (config as any).deterministicChainMode = false; // reset
      const afterHidden = net.nodes.filter((n) => n.type === 'hidden').length;
      // Assert
      expect(afterHidden - beforeHidden).toBe(1);
    });
  });

  describe('Scenario: ADD_SELF_CONN blocked by acyclicity', () => {
    it('does not create self connection when enforceAcyclic=true', () => {
      // Arrange
      const net = new Network(2, 1, { seed: 3, enforceAcyclic: true });
      const before = net.selfconns.length;
      // Act
      mutateImpl.call(net, mutation.ADD_SELF_CONN);
      // Assert
      expect(net.selfconns.length).toBe(before);
    });
  });

  describe('Scenario: ADD_BACK_CONN blocked by acyclicity', () => {
    it('does not add backward connection when enforceAcyclic=true', () => {
      // Arrange
      const net = new Network(2, 2, { seed: 4, enforceAcyclic: true });
      const before = net.connections.length;
      // Act
      mutateImpl.call(net, mutation.ADD_BACK_CONN);
      // Assert
      expect(net.connections.length).toBe(before);
    });
  });

  describe('Scenario: ADD_GATE then SUB_GATE', () => {
    it('increments then decrements gate count', () => {
      // Arrange
      const net = new Network(2, 2, { seed: 5 });
      mutateImpl.call(net, mutation.ADD_GATE);
      const afterAdd = net.gates.length;
      mutateImpl.call(net, mutation.SUB_GATE);
      const afterSub = net.gates.length;
      // Assert
      expect(afterAdd > afterSub).toBe(true);
    });
  });

  describe('Scenario: MOD_ACTIVATION excluding outputs when only outputs exist', () => {
    it('no-ops when mutateOutput=false and no hidden nodes', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 6 });
      const before = net.nodes.map((n) => n.squash.name).join(',');
      // Act
      mutateImpl.call(net, { ...mutation.MOD_ACTIVATION, mutateOutput: false });
      const after = net.nodes.map((n) => n.squash.name).join(',');
      // Assert
      expect(after).toBe(before);
    });
  });

  describe('Scenario: SWAP_NODES excluding outputs (insufficient swappables)', () => {
    it('leaves network unchanged when <2 swappable nodes', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 7 });
      const before = net.nodes.map((n) => n.bias).join(',');
      // Act
      mutateImpl.call(net, { ...mutation.SWAP_NODES, mutateOutput: false });
      const after = net.nodes.map((n) => n.bias).join(',');
      // Assert
      expect(after).toBe(before);
    });
  });
});
