/**
 * network.slab.fast.gating.guard.test.ts
 * Validates that when gating is present, fastSlabActivate falls back to legacy path (guard).
 */
import Network from '../../src/architecture/network';
import { config } from '../../src/config';
import Node from '../../src/architecture/node';

// Minimal gating scenario: create a gater connection (simulate by assigning to net.gates array directly if API limited)

describe('network.slab.fast.gating.guard', () => {
  it('falls back to legacy path when gating detected', () => {
    // Arrange
    config.enableNodePooling = false;
    const net = new Network(2, 1, { enforceAcyclic: true });
    // Create a hidden node and a dummy gated connection by pushing to gates array (simplified)
    const hidden = new Node('hidden');
    (net as any).nodes.push(hidden);
    (net as any).gates.push({ dummy: true }); // simulate gating presence
    (net as any)._topoDirty = true;
    (net as any)._nodeIndexDirty = true;
    const input = [0.3, -0.1];
    const legacy = (net as any).activate(input.slice(), false);
    const fast = (net as any).fastSlabActivate(input.slice());
    // Assert: outputs equal (fallback used) and gating array present
    expect(
      JSON.stringify(fast) === JSON.stringify(legacy) &&
        (net as any).gates.length > 0
    ).toBe(true);
  });
});
