/**
 * @module routing-table.red.test
 * @description Red tests for the query routing table and classifyAndRoute function.
 *
 * Uses runModuleEvaluation to call .mjs module functions from .ts test files,
 * matching the established pattern in semantic-index.red.test.ts.
 */
import { execFileSync } from 'node:child_process';
import path from 'node:path';

// ---------------------------------------------------------------------------
// Types for routing results
// ---------------------------------------------------------------------------

interface RoutingStrategy {
  family: string | null;
  expansion: string;
  post_processing: string;
}

interface ClassifyAndRouteResult {
  query_class: string;
  confidence: number;
  alpha: number;
  strategy: RoutingStrategy;
}

// ---------------------------------------------------------------------------
// Helper: evaluate .mjs modules via subprocess (matching established pattern)
// ---------------------------------------------------------------------------
const runModuleEvaluation = <Result>(source: string): Result => {
  const output = execFileSync(
    process.execPath,
    ['--input-type=module', '--eval', source],
    {
      cwd: path.resolve(process.cwd()),
      encoding: 'utf8',
    },
  );
  return JSON.parse(output) as Result;
};

// ---------------------------------------------------------------------------
// DEFAULTS — canonical alpha values
// ---------------------------------------------------------------------------

describe('routing-table.mjs', () => {
  describe('DEFAULTS', () => {
    it('contains all six query classes with correct alpha values', () => {
      const defaults = runModuleEvaluation<Record<string, number>>(`
        import { DEFAULTS } from './rag-index/routing-table.mjs';
        console.log(JSON.stringify(DEFAULTS));
      `);
      expect(defaults).toEqual({
        simple_lookup: 0.75,
        cross_boundary: 0.5,
        multi_hop: 0.35,
        exploratory: 0.3,
        code_specific: 0.7,
        plan_specific: 0.7,
      });
    });

    it('is frozen and cannot be mutated', () => {
      const result = runModuleEvaluation<boolean>(`
        import { DEFAULTS } from './rag-index/routing-table.mjs';
        console.log(JSON.stringify(Object.isFrozen(DEFAULTS)));
      `);
      expect(result).toBe(true);
    });
  });

  // ---------------------------------------------------------------------------
  // ROUTING — canonical routing strategies
  // ---------------------------------------------------------------------------

  describe('ROUTING', () => {
    it('contains all six query classes with correct strategies', () => {
      const routing = runModuleEvaluation<Record<string, RoutingStrategy>>(`
        import { ROUTING } from './rag-index/routing-table.mjs';
        console.log(JSON.stringify(ROUTING));
      `);
      expect(routing).toEqual({
        simple_lookup: {
          family: null,
          expansion: 'none',
          post_processing: 'default',
        },
        cross_boundary: {
          family: null,
          expansion: 'multi_family',
          post_processing: 'cross_family_dedup',
        },
        multi_hop: {
          family: null,
          expansion: 'entity_graph',
          post_processing: 'hop_decay',
        },
        exploratory: {
          family: null,
          expansion: 'context_assembly',
          post_processing: 'budget_assembly',
        },
        code_specific: {
          family: 'ts-source',
          expansion: 'none',
          post_processing: 'default',
        },
        plan_specific: {
          family: 'plan,completed-plan',
          expansion: 'none',
          post_processing: 'default',
        },
      });
    });

    it('is frozen and cannot be mutated', () => {
      const result = runModuleEvaluation<boolean>(`
        import { ROUTING } from './rag-index/routing-table.mjs';
        console.log(JSON.stringify(Object.isFrozen(ROUTING)));
      `);
      expect(result).toBe(true);
    });
  });

  // ---------------------------------------------------------------------------
  // classifyAndRoute — full routing function
  // ---------------------------------------------------------------------------

  describe('classifyAndRoute', () => {
    it('returns routing for code_specific queries with dotted identifiers', () => {
      const result = runModuleEvaluation<ClassifyAndRouteResult>(`
        import { classifyAndRoute } from './rag-index/routing-table.mjs';
        console.log(JSON.stringify(classifyAndRoute('Network.activate')));
      `);
      expect(result.query_class).toBe('code_specific');
      expect(result.confidence).toBe(0.85);
      expect(result.alpha).toBe(0.7);
      expect(result.strategy.family).toBe('ts-source');
      expect(result.strategy.expansion).toBe('none');
      expect(result.strategy.post_processing).toBe('default');
    });

    it('returns routing for exploratory queries', () => {
      const result = runModuleEvaluation<ClassifyAndRouteResult>(`
        import { classifyAndRoute } from './rag-index/routing-table.mjs';
        console.log(JSON.stringify(classifyAndRoute('how does the training pipeline work')));
      `);
      expect(result.query_class).toBe('exploratory');
      expect(result.alpha).toBe(0.3);
      expect(result.strategy.family).toBeNull();
      expect(result.strategy.expansion).toBe('context_assembly');
    });

    it('returns routing for code_specific queries', () => {
      const result = runModuleEvaluation<ClassifyAndRouteResult>(`
        import { classifyAndRoute } from './rag-index/routing-table.mjs';
        console.log(JSON.stringify(classifyAndRoute('implementation of crossover in NEAT')));
      `);
      expect(result.query_class).toBe('code_specific');
      expect(result.alpha).toBe(0.7);
      expect(result.strategy.family).toBe('ts-source');
    });

    it('returns routing for plan_specific queries', () => {
      const result = runModuleEvaluation<ClassifyAndRouteResult>(`
        import { classifyAndRoute } from './rag-index/routing-table.mjs';
        console.log(JSON.stringify(classifyAndRoute('what is the checkpointing design')));
      `);
      expect(result.query_class).toBe('plan_specific');
      expect(result.alpha).toBe(0.7);
      expect(result.strategy.family).toBe('plan,completed-plan');
    });

    it('returns routing for multi_hop queries', () => {
      const result = runModuleEvaluation<ClassifyAndRouteResult>(`
        import { classifyAndRoute } from './rag-index/routing-table.mjs';
        console.log(JSON.stringify(classifyAndRoute('functions that also call activate')));
      `);
      expect(result.query_class).toBe('multi_hop');
      expect(result.alpha).toBe(0.35);
      expect(result.strategy.expansion).toBe('entity_graph');
    });

    it('returns routing for cross_boundary queries', () => {
      const result = runModuleEvaluation<ClassifyAndRouteResult>(`
        import { classifyAndRoute } from './rag-index/routing-table.mjs';
        console.log(JSON.stringify(classifyAndRoute('relationship between crossover and mutation')));
      `);
      expect(result.query_class).toBe('cross_boundary');
      expect(result.alpha).toBe(0.5);
      expect(result.strategy.expansion).toBe('multi_family');
    });

    // -------------------------------------------------------------------------
    // classification_hints overrides
    // -------------------------------------------------------------------------

    describe('classification_hints', () => {
      it('allows overriding alpha via classification_hints', () => {
        const result = runModuleEvaluation<ClassifyAndRouteResult>(`
          import { classifyAndRoute } from './rag-index/routing-table.mjs';
          console.log(JSON.stringify(classifyAndRoute('Network.activate', { alpha: 0.9 })));
        `);
        expect(result.alpha).toBe(0.9);
        expect(result.query_class).toBe('code_specific');
      });

      it('allows overriding family via classification_hints', () => {
        const result = runModuleEvaluation<ClassifyAndRouteResult>(`
          import { classifyAndRoute } from './rag-index/routing-table.mjs';
          console.log(JSON.stringify(classifyAndRoute('Network.activate', { family: 'plan' })));
        `);
        expect(result.strategy.family).toBe('plan');
      });

      it('uses per-class default alpha when no hints are provided', () => {
        const result = runModuleEvaluation<ClassifyAndRouteResult>(`
          import { classifyAndRoute } from './rag-index/routing-table.mjs';
          console.log(JSON.stringify(classifyAndRoute('Network.activate')));
        `);
        expect(result.alpha).toBe(0.7);
      });

      it('uses per-class default strategy when no hints are provided', () => {
        const result = runModuleEvaluation<ClassifyAndRouteResult>(`
          import { classifyAndRoute } from './rag-index/routing-table.mjs';
          console.log(JSON.stringify(classifyAndRoute('Network.activate')));
        `);
        expect(result.strategy.family).toBe('ts-source');
        expect(result.strategy.expansion).toBe('none');
      });
    });
  });
});
