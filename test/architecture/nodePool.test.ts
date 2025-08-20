import {
  acquireNode,
  releaseNode,
  nodePoolStats,
  resetNodePool,
} from '../../src/architecture/nodePool';

/**
 * NodePool skeleton tests (Phase 1 groundwork)
 * Ensures acquire/release reset semantics & isolation.
 */

describe('NodePool skeleton', () => {
  afterEach(() => {
    resetNodePool();
  });

  describe('acquireNode()', () => {
    const n = acquireNode({ type: 'hidden' });
    it('provides node with zero activation', () => {
      expect(n.activation).toBe(0);
    });
  });

  describe('release + acquire cycle', () => {
    const first = acquireNode({ type: 'hidden' });
    first.activation = 42; // mutate state
    releaseNode(first);
    const second = acquireNode({ type: 'output' });
    it('resets activation to zero on recycled node', () => {
      expect(second.activation).toBe(0);
    });
  });

  describe('pool stats reflect releases', () => {
    const a = acquireNode();
    releaseNode(a);
    const stats = nodePoolStats();
    it('shows size=1 after single release', () => {
      expect(stats.size).toBe(1);
    });
  });

  describe('acquire respects requested type', () => {
    const n = acquireNode({ type: 'input' });
    it('returns node with requested type', () => {
      expect(n.type).toBe('input');
    });
  });
});
