import { describe, expect, it } from '@jest/globals';

import type * as MainAgent from './main-agent';

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/harness/main-agent.ts.
 *
 * Covers the Phase 4 NGE main-agent harness integration:
 * - AC-401: lifecycle stages advance through Embryo -> Juvenile -> Adult -> Reproducing.
 * - AC-402: tier-capped topology is enforced at every lifecycle transition.
 * - AC-403: motif set is exactly the existing catalogue allowlist.
 */

interface MainAgentModule {
  runMainAgentGeneration: typeof MainAgent.runMainAgentGeneration;
  NeatensteinMainAgentTierBudget: typeof MainAgent.NeatensteinMainAgentTierBudget;
  NeatensteinMainAgentMotifAllowlist: typeof MainAgent.NeatensteinMainAgentMotifAllowlist;
}

describe('Neatenstein harness main-agent', () => {
  describe('AC-401: lifecycle runner contract', () => {
    it('exports runMainAgentGeneration as a function', async () => {
      const mod = (await import('./main-agent.ts')) as Record<string, unknown>;
      expect(typeof mod.runMainAgentGeneration).toBe('function');
    });

    it('returns a CombatQualitySignal for a valid generation', async () => {
      const { runMainAgentGeneration } =
        (await import('./main-agent.ts')) as MainAgentModule;
      const result = runMainAgentGeneration({ seed: 1, generation: 0 });
      expect({
        hasSurvivalTicks: typeof result.survivalTicks === 'number',
        hasDamageDealt: typeof result.damageDealt === 'number',
        hasKills: typeof result.kills === 'number',
        hasDamageTaken: typeof result.damageTaken === 'number',
        hasAimMissRate: typeof result.aimMissRate === 'number',
        hasComplexityBonus: typeof result.complexityBonus === 'number',
        hasParsimonyDensityPenalty:
          typeof result.parsimonyDensityPenalty === 'number',
      }).toEqual({
        hasSurvivalTicks: true,
        hasDamageDealt: true,
        hasKills: true,
        hasDamageTaken: true,
        hasAimMissRate: true,
        hasComplexityBonus: true,
        hasParsimonyDensityPenalty: true,
      });
    });

    it('advances the lifecycle stage from the previous generation', async () => {
      const { runMainAgentGeneration } =
        (await import('./main-agent.ts')) as MainAgentModule;
      const first = runMainAgentGeneration({ seed: 1, generation: 0 });
      const second = runMainAgentGeneration({ seed: 1, generation: 1 });
      expect(second.stage).not.toEqual(first.stage);
    });

    it('produces deterministic output for the same config', async () => {
      const { runMainAgentGeneration } =
        (await import('./main-agent.ts')) as MainAgentModule;
      const config = { seed: 7, generation: 2 };
      const first = runMainAgentGeneration(config);
      const second = runMainAgentGeneration(config);
      expect(first).toEqual(second);
    });

    it('accepts a frozen enemy snapshot for evaluation', async () => {
      const { runMainAgentGeneration } =
        (await import('./main-agent.ts')) as MainAgentModule;
      const result = runMainAgentGeneration({
        seed: 3,
        generation: 1,
        enemySnapshot: { kind: 'mlp', weights: new Float32Array(80) },
      });
      expect(typeof result.survivalTicks).toBe('number');
    });
  });

  describe('AC-402 + AC-403: topology and motif constraints', () => {
    it('exports a tier budget for the main agent', async () => {
      const mod = (await import('./main-agent.ts')) as Record<string, unknown>;
      expect(typeof mod.NeatensteinMainAgentTierBudget).toBe('object');
    });

    it('exports the allowed motif catalogue', async () => {
      const mod = (await import('./main-agent.ts')) as Record<string, unknown>;
      expect(Array.isArray(mod.NeatensteinMainAgentMotifAllowlist)).toBe(true);
    });
  });

  describe('AC-401-S02-001: real NGE main-agent genome contract', () => {
    it('returns a champion genome built by the NGE pipeline', async () => {
      const { runMainAgentGeneration } =
        (await import('./main-agent.ts')) as MainAgentModule;
      const result = runMainAgentGeneration({ seed: 42, generation: 0 });
      expect(
        (result as unknown as Record<string, unknown>).championGenome,
      ).toBeDefined();
    });

    it('does not return a placeholder genome of null arrays', async () => {
      const { runMainAgentGeneration } =
        (await import('./main-agent.ts')) as MainAgentModule;
      const result = runMainAgentGeneration({ seed: 42, generation: 0 });
      const extendedResult = result as unknown as Record<string, unknown>;
      const genome = extendedResult.championGenome as
        { nodes?: unknown[]; archetypes?: unknown[] } | undefined;
      expect(genome?.archetypes?.length).toBeGreaterThan(0);
    });
  });

  describe('AC-401-S02-002: tier-capped topology and motif allowlist', () => {
    it('caps the champion genome node count at the tier budget', async () => {
      const { runMainAgentGeneration, NeatensteinMainAgentTierBudget } =
        (await import('./main-agent.ts')) as MainAgentModule;
      const result = runMainAgentGeneration({ seed: 42, generation: 0 });
      const extendedResult = result as unknown as Record<string, unknown>;
      const genome = extendedResult.championGenome as
        { nodeCount?: number } | undefined;
      expect(genome?.nodeCount ?? Infinity).toBeLessThanOrEqual(
        NeatensteinMainAgentTierBudget.maxNodes,
      );
    });

    it('caps the champion genome edge count at the tier budget', async () => {
      const { runMainAgentGeneration, NeatensteinMainAgentTierBudget } =
        (await import('./main-agent.ts')) as MainAgentModule;
      const result = runMainAgentGeneration({ seed: 42, generation: 0 });
      const extendedResult = result as unknown as Record<string, unknown>;
      const genome = extendedResult.championGenome as
        { edgeCount?: number } | undefined;
      expect(genome?.edgeCount ?? Infinity).toBeLessThanOrEqual(
        NeatensteinMainAgentTierBudget.maxEdges,
      );
    });

    it('materializes exactly the three allowlisted motif archetypes', async () => {
      const { runMainAgentGeneration, NeatensteinMainAgentMotifAllowlist } =
        (await import('./main-agent.ts')) as MainAgentModule;
      const result = runMainAgentGeneration({ seed: 42, generation: 0 });
      const extendedResult = result as unknown as Record<string, unknown>;
      const genome = extendedResult.championGenome as
        { archetypes?: { computationType: string }[] } | undefined;
      const allowlist = [...NeatensteinMainAgentMotifAllowlist].sort();
      const archetypes = genome?.archetypes ?? [];
      const motifKinds = archetypes
        .map((archetype) => archetype.computationType)
        .sort();
      expect(motifKinds).toEqual(allowlist);
    });
  });

  describe('AC-401-S02-003: deterministic lifecycle progression', () => {
    it('carries a full NGE lifecycle state beyond the stage label', async () => {
      const { runMainAgentGeneration } =
        (await import('./main-agent.ts')) as MainAgentModule;
      const result = runMainAgentGeneration({ seed: 42, generation: 0 });
      expect(
        (result as unknown as Record<string, unknown>).lifecycleState,
      ).toBeDefined();
    });

    it('progresses through embryo, juvenile, adult, and reproducing over four generations', async () => {
      const { runMainAgentGeneration } =
        (await import('./main-agent.ts')) as MainAgentModule;
      const stages = [0, 1, 2, 3].map((generation) => {
        const result = runMainAgentGeneration({ seed: 42, generation });
        const lifecycleState = (result as unknown as Record<string, unknown>)
          .lifecycleState as { stage?: string } | undefined;
        return lifecycleState?.stage;
      });
      expect(stages).toEqual(['embryo', 'juvenile', 'adult', 'reproducing']);
    });
  });

  describe('AC-401-S02-005: assimilation writes only internal priors', () => {
    it('reports no enemy-derived weights in the assimilation result', async () => {
      const { runMainAgentGeneration } =
        (await import('./main-agent.ts')) as MainAgentModule;
      const result = runMainAgentGeneration({ seed: 42, generation: 2 });
      const assimilation = (result as unknown as Record<string, unknown>)
        .assimilation as { enemyWeightsIncorporated?: boolean } | undefined;
      expect(assimilation?.enemyWeightsIncorporated).toBe(false);
    });
  });
});
