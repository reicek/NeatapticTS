/**
 * Red-phase contracts for capacity limits (Phase 4 Step 14).
 *
 * These tests verify that:
 * - `DEFAULT_LIMITS.maxNodes` is raised to 8000
 * - `DEFAULT_LIMITS.maxConnections` is raised to 32000
 * - NGE growth budget constants support 8000 nodes / 32000 connections
 * - Throttling prevents excessive lifecycle runs when the network is large
 *
 * They will fail until Phase 4 Step 15 is implemented.
 *
 * The tests use source-text inspection (the same pattern as
 * `runtime.adaptation.lifecycle.test.ts`) because `DEFAULT_LIMITS` is a
 * module-private constant not exported from the module surface.
 *
 * Single-expect rule enforced throughout. AAA structure in every test.
 */
import * as fs from 'node:fs';
import * as path from 'node:path';

const ADAPTATION_SOURCE = path.resolve(__dirname, 'runtime.adaptation.ts');

const CONSTANTS_SOURCE = path.resolve(
  __dirname,
  '../../../src/neat/nge-juvenile/neat.nge-juvenile.constants.ts',
);

function readAdaptationSource(): string {
  return fs.readFileSync(ADAPTATION_SOURCE, 'utf-8');
}

function readConstantsSource(): string {
  return fs.readFileSync(CONSTANTS_SOURCE, 'utf-8');
}

/**
 * Extract the `DEFAULT_LIMITS` object literal from the adaptation source text.
 * Returns the raw text of the object body or an empty string if not found.
 */
function extractDefaultLimitsBlock(sourceText: string): string {
  const match = /DEFAULT_LIMITS[^{]*\{[^}]*\}/.exec(sourceText);
  return match ? match[0] : '';
}

describe('capacity limits (Phase 4 Step 14)', () => {
  describe('DEFAULT_LIMITS values', () => {
    it('maxNodes is 8000', () => {
      const sourceText = readAdaptationSource();
      const limitsBlock = extractDefaultLimitsBlock(sourceText);

      const maxNodesMatch = /maxNodes:\s*(\d[\d_]*)/.exec(limitsBlock);
      const maxNodes = maxNodesMatch
        ? parseInt(maxNodesMatch[1].replace(/_/g, ''), 10)
        : 0;

      expect(maxNodes).toBe(8000);
    });

    it('maxConnections is 32000', () => {
      const sourceText = readAdaptationSource();
      const limitsBlock = extractDefaultLimitsBlock(sourceText);

      const maxConnectionsMatch = /maxConnections:\s*(\d[\d_]*)/.exec(
        limitsBlock,
      );
      const maxConnections = maxConnectionsMatch
        ? parseInt(maxConnectionsMatch[1].replace(/_/g, ''), 10)
        : 0;

      expect(maxConnections).toBe(32000);
    });
  });

  describe('NGE growth budget constants', () => {
    it('constants file exports a max node capacity of 8000', () => {
      const sourceText = readConstantsSource();

      // Look for a constant name containing MAX and NODE(S) with value 8000.
      const hasMaxNodesConstant =
        /NGE_[A-Z_]*MAX[A-Z_]*NODE[A-Z_]*=\s*8[\s_]*000/.test(sourceText);

      expect(hasMaxNodesConstant).toBe(true);
    });

    it('constants file exports a max edge capacity of 32000', () => {
      const sourceText = readConstantsSource();

      // Look for a constant name containing MAX and EDGE(S) with value 32000.
      const hasMaxEdgesConstant =
        /NGE_[A-Z_]*MAX[A-Z_]*EDGE[A-Z_]*=\s*32[\s_]*000/.test(sourceText);

      expect(hasMaxEdgesConstant).toBe(true);
    });
  });

  describe('growth throttling', () => {
    it('adaptOnTick has size-based throttling before runNgeLifecycle call', () => {
      const sourceText = readAdaptationSource();

      // Find the runNgeLifecycle call and inspect the code before it
      // for a size-based throttle mechanism.
      const lifecycleCallIndex = sourceText.indexOf('runNgeLifecycle(');
      const beforeLifecycle =
        lifecycleCallIndex > 0
          ? sourceText.substring(0, lifecycleCallIndex)
          : '';

      // After the cooldown/rollback guards but before the lifecycle call,
      // there should be a throttle check that considers network size.
      const hasSizeThrottle =
        /throttl|sizeThreshold|largeNetwork|backOff|growthThrottle|sizeBudget/i.test(
          beforeLifecycle,
        );

      expect(hasSizeThrottle).toBe(true);
    });
  });
});
