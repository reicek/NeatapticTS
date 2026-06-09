/**
 * @module extract-cross-refs.red.test
 * @description Red tests for cross-reference extraction from document body text.
 */
describe('extract-cross-refs', () => {
  describe('extractCrossRefs', () => {
    it('extracts references from src/<path>/<file>.ts patterns', async () => {
      const { extractCrossRefs } = await import('../extract-cross-refs.mjs');

      const docEntity = {
        entity_type: 'plan',
        qualified_name: 'plans/test_plan',
        file_path: 'plans/test_plan.plans.md',
      };

      const codeEntityMap = new Map([
        [
          'src/architecture/network',
          {
            entity_type: 'module',
            name: 'network',
            qualified_name: 'src/architecture/network',
          },
        ],
      ]);

      const docEntityMap = new Map();
      const headingsByDoc = new Map([
        [
          'plans/test_plan',
          ['See src/architecture/network/network.ts for the implementation.'],
        ],
      ]);

      const result = extractCrossRefs({
        docEntities: [docEntity],
        codeEntityMap,
        docEntityMap,
        headingsByDoc,
      });

      const srcRefEdges = result.edges.filter(
        (e) => e.target_qualified_name === 'src/architecture/network',
      );
      expect(srcRefEdges.length).toBeGreaterThan(0);
      expect(srcRefEdges[0].confidence).toBe('medium');
    });

    it('extracts references from ClassName.methodName patterns', async () => {
      const { extractCrossRefs } = await import('../extract-cross-refs.mjs');

      const docEntity = {
        entity_type: 'plan',
        qualified_name: 'plans/test_plan',
      };

      const codeEntityMap = new Map([
        [
          'src/architecture/network.Network.evolve',
          {
            entity_type: 'function',
            name: 'evolve',
            qualified_name: 'src/architecture/network.Network.evolve',
          },
        ],
      ]);

      const docEntityMap = new Map();
      const headingsByDoc = new Map([
        ['plans/test_plan', ['Use Network.evolve to train the network.']],
      ]);

      const result = extractCrossRefs({
        docEntities: [docEntity],
        codeEntityMap,
        docEntityMap,
        headingsByDoc,
      });

      const methodRefEdges = result.edges.filter(
        (e) =>
          e.target_qualified_name === 'src/architecture/network.Network.evolve',
      );
      expect(methodRefEdges.length).toBeGreaterThan(0);
      expect(methodRefEdges[0].confidence).toBe('medium');
    });

    it('extracts references from ClassName patterns with low confidence', async () => {
      const { extractCrossRefs } = await import('../extract-cross-refs.mjs');

      const docEntity = {
        entity_type: 'plan',
        qualified_name: 'plans/test_plan',
      };

      const codeEntityMap = new Map([
        [
          'src/architecture/network.Network',
          {
            entity_type: 'class',
            name: 'Network',
            qualified_name: 'src/architecture/network.Network',
          },
        ],
      ]);

      const docEntityMap = new Map();
      const headingsByDoc = new Map([
        [
          'plans/test_plan',
          ['The Network class provides the core functionality.'],
        ],
      ]);

      const result = extractCrossRefs({
        docEntities: [docEntity],
        codeEntityMap,
        docEntityMap,
        headingsByDoc,
      });

      const classRefEdges = result.edges.filter(
        (e) => e.target_qualified_name === 'src/architecture/network.Network',
      );
      expect(classRefEdges.length).toBeGreaterThan(0);
      expect(classRefEdges[0].confidence).toBe('low');
    });

    it('extracts references from backtick-quoted code symbols', async () => {
      const { extractCrossRefs } = await import('../extract-cross-refs.mjs');

      const docEntity = {
        entity_type: 'plan',
        qualified_name: 'plans/test_plan',
      };

      const codeEntityMap = new Map([
        [
          'src/methods.selection.tournamentSelection',
          {
            entity_type: 'function',
            name: 'tournamentSelection',
            qualified_name: 'src/methods.selection.tournamentSelection',
          },
        ],
      ]);

      const docEntityMap = new Map();
      const headingsByDoc = new Map([
        [
          'plans/test_plan',
          ['Use `tournamentSelection` for the selection method.'],
        ],
      ]);

      const result = extractCrossRefs({
        docEntities: [docEntity],
        codeEntityMap,
        docEntityMap,
        headingsByDoc,
      });

      const backtickEdges = result.edges.filter(
        (e) =>
          e.target_qualified_name ===
          'src/methods.selection.tournamentSelection',
      );
      expect(backtickEdges.length).toBeGreaterThan(0);
      expect(backtickEdges[0].confidence).toBe('low');
    });

    it('extracts references from plans/<plan-name> patterns', async () => {
      const { extractCrossRefs } = await import('../extract-cross-refs.mjs');

      const docEntity = {
        entity_type: 'agent',
        qualified_name: 'agents/04-implementing',
      };

      const codeEntityMap = new Map();
      const docEntityMap = new Map([
        [
          'plans/repo_cortex',
          {
            entity_type: 'plan',
            name: 'repo_cortex',
            qualified_name: 'plans/repo_cortex',
          },
        ],
      ]);

      const headingsByDoc = new Map([
        [
          'agents/04-implementing',
          ['See plans/repo_cortex for the full design.'],
        ],
      ]);

      const result = extractCrossRefs({
        docEntities: [docEntity],
        codeEntityMap,
        docEntityMap,
        headingsByDoc,
      });

      const planRefEdges = result.edges.filter(
        (e) => e.target_qualified_name === 'plans/repo_cortex',
      );
      expect(planRefEdges.length).toBeGreaterThan(0);
      expect(planRefEdges[0].confidence).toBe('medium');
    });

    it('extracts references from .github/skills/<skill-name> patterns', async () => {
      const { extractCrossRefs } = await import('../extract-cross-refs.mjs');

      const docEntity = {
        entity_type: 'agent',
        qualified_name: 'agents/04-implementing',
      };

      const codeEntityMap = new Map();
      const docEntityMap = new Map([
        [
          'skills/coverage-guard',
          {
            entity_type: 'skill',
            name: 'coverage-guard',
            qualified_name: 'skills/coverage-guard',
          },
        ],
      ]);

      const headingsByDoc = new Map([
        [
          'agents/04-implementing',
          ['Delegate to .github/skills/coverage-guard for coverage checks.'],
        ],
      ]);

      const result = extractCrossRefs({
        docEntities: [docEntity],
        codeEntityMap,
        docEntityMap,
        headingsByDoc,
      });

      const skillRefEdges = result.edges.filter(
        (e) => e.target_qualified_name === 'skills/coverage-guard',
      );
      expect(skillRefEdges.length).toBeGreaterThan(0);
      expect(skillRefEdges[0].confidence).toBe('medium');
    });

    it('deduplicates edges between same source and target', async () => {
      const { extractCrossRefs } = await import('../extract-cross-refs.mjs');

      const docEntity = {
        entity_type: 'plan',
        qualified_name: 'plans/test_plan',
      };

      const codeEntityMap = new Map([
        [
          'src/architecture/network',
          {
            entity_type: 'module',
            name: 'network',
            qualified_name: 'src/architecture/network',
          },
        ],
      ]);

      const docEntityMap = new Map();
      const headingsByDoc = new Map([
        [
          'plans/test_plan',
          [
            'See src/architecture/network/network.ts for details.',
            'Also see src/architecture/network/network.ts for the API.',
          ],
        ],
      ]);

      const result = extractCrossRefs({
        docEntities: [docEntity],
        codeEntityMap,
        docEntityMap,
        headingsByDoc,
      });

      // Should only have one edge for the same source→target pair.
      const networkEdges = result.edges.filter(
        (e) => e.target_qualified_name === 'src/architecture/network',
      );
      expect(networkEdges.length).toBe(1);
    });

    it('skips common non-class words for ClassName pattern', async () => {
      const { extractCrossRefs } = await import('../extract-cross-refs.mjs');

      const docEntity = {
        entity_type: 'plan',
        qualified_name: 'plans/test_plan',
      };

      const codeEntityMap = new Map([
        [
          'src/utils.The',
          {
            entity_type: 'class',
            name: 'The',
            qualified_name: 'src/utils.The',
          },
        ],
      ]);

      const docEntityMap = new Map();
      const headingsByDoc = new Map([
        ['plans/test_plan', ['The implementation uses the network.']],
      ]);

      const result = extractCrossRefs({
        docEntities: [docEntity],
        codeEntityMap,
        docEntityMap,
        headingsByDoc,
      });

      // "The" should be skipped as a common non-class word.
      const theEdges = result.edges.filter(
        (e) => e.target_qualified_name === 'src/utils.The',
      );
      expect(theEdges.length).toBe(0);
    });

    it('limits scanning to first 2000 chars per heading section', async () => {
      const { extractCrossRefs } = await import('../extract-cross-refs.mjs');

      const docEntity = {
        entity_type: 'plan',
        qualified_name: 'plans/test_plan',
      };

      const codeEntityMap = new Map([
        [
          'src/architecture/network',
          {
            entity_type: 'module',
            name: 'network',
            qualified_name: 'src/architecture/network',
          },
        ],
      ]);

      const docEntityMap = new Map();

      // Create a heading section > 2000 chars with the reference near the end.
      const padding = 'x'.repeat(2500);
      const longText = `${padding} See src/architecture/network/network.ts for details.`;

      const headingsByDoc = new Map([['plans/test_plan', [longText]]]);

      const result = extractCrossRefs({
        docEntities: [docEntity],
        codeEntityMap,
        docEntityMap,
        headingsByDoc,
      });

      // The reference after 2000 chars should NOT be found.
      const networkEdges = result.edges.filter(
        (e) => e.target_qualified_name === 'src/architecture/network',
      );
      expect(networkEdges.length).toBe(0);
    });

    it('produces only references edges', async () => {
      const { extractCrossRefs } = await import('../extract-cross-refs.mjs');

      const docEntity = {
        entity_type: 'plan',
        qualified_name: 'plans/test_plan',
      };

      const codeEntityMap = new Map([
        [
          'src/architecture/network',
          {
            entity_type: 'module',
            name: 'network',
            qualified_name: 'src/architecture/network',
          },
        ],
      ]);

      const docEntityMap = new Map();
      const headingsByDoc = new Map([
        ['plans/test_plan', ['See src/architecture/network/network.ts.']],
      ]);

      const result = extractCrossRefs({
        docEntities: [docEntity],
        codeEntityMap,
        docEntityMap,
        headingsByDoc,
      });

      for (const edge of result.edges) {
        expect(edge.relationship).toBe('references');
      }
    });
  });
});
