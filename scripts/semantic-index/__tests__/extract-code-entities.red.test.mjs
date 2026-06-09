/**
 * @module extract-code-entities.red.test
 * @description Red tests for code entity extraction from TypeScript source files
 * using ts-morph AST analysis. Pure functions are tested directly; full extraction
 * tests use limited source scope to keep execution time reasonable.
 */
import path from 'node:path';

describe('extract-code-entities', () => {
  describe('deriveModulePath', () => {
    it('derives module path from root-level .ts files (src/neat.ts → src/neat)', async () => {
      const { deriveModulePath } = await import('../extract-code-entities.mjs');
      expect(deriveModulePath('src/neat.ts')).toBe('src/neat');
    });

    it('derives module path from nested barrel files (src/architecture/network/network.ts → src/architecture/network/network)', async () => {
      const { deriveModulePath } = await import('../extract-code-entities.mjs');
      expect(deriveModulePath('src/architecture/network/network.ts')).toBe(
        'src/architecture/network/network',
      );
    });

    it('derives module path from deeply nested non-barrel files', async () => {
      const { deriveModulePath } = await import('../extract-code-entities.mjs');
      expect(
        deriveModulePath(
          'src/architecture/network/worker-payload/network.worker-payload.shared.ts',
        ),
      ).toBe(
        'src/architecture/network/worker-payload/network.worker-payload.shared',
      );
    });

    it('derives module path from non-barrel nested files (src/architecture/architect.ts → src/architecture/architect)', async () => {
      const { deriveModulePath } = await import('../extract-code-entities.mjs');
      expect(deriveModulePath('src/architecture/architect.ts')).toBe(
        'src/architecture/architect',
      );
    });

    it('handles index.ts files in nested directories', async () => {
      const { deriveModulePath } = await import('../extract-code-entities.mjs');
      expect(deriveModulePath('src/methods/selection/index.ts')).toBe(
        'src/methods/selection/index',
      );
    });

    it('handles root-level files with underscores', async () => {
      const { deriveModulePath } = await import('../extract-code-entities.mjs');
      expect(deriveModulePath('src/crossover_method.ts')).toBe(
        'src/crossover_method',
      );
    });
  });

  describe('extractCodeEntities (limited scope)', () => {
    it('extracts module entities with correct qualified_name = module_path', async () => {
      const { extractCodeEntities } =
        await import('../extract-code-entities.mjs');
      // Limit to a small source file to keep test fast.
      const result = await extractCodeEntities({
        sourcePaths: [path.resolve('src/neat.ts')],
      });

      const moduleEntities = result.entities.filter(
        (e) => e.entity_type === 'module',
      );
      expect(moduleEntities.length).toBeGreaterThan(0);

      for (const mod of moduleEntities) {
        expect(mod.qualified_name).toBe(mod.module_path);
        expect(mod.entity_type).toBe('module');
      }
    });

    it('extracts class entities with correct qualified_name pattern (module.ClassName)', async () => {
      const { extractCodeEntities } =
        await import('../extract-code-entities.mjs');
      const result = await extractCodeEntities({
        sourcePaths: [path.resolve('src/architecture/network/network.ts')],
      });

      const classEntities = result.entities.filter(
        (e) => e.entity_type === 'class',
      );
      for (const cls of classEntities) {
        // Qualified name follows module.ClassName; `default` exports are lowercase.
        const parts = cls.qualified_name.split('.');
        expect(parts.length).toBeGreaterThanOrEqual(2);
      }
    });

    it('extracts owns edges from modules to symbols', async () => {
      const { extractCodeEntities } =
        await import('../extract-code-entities.mjs');
      const result = await extractCodeEntities({
        sourcePaths: [path.resolve('src/architecture/network/network.ts')],
      });

      const ownsEdges = result.edges.filter((e) => e.relationship === 'owns');
      expect(ownsEdges.length).toBeGreaterThan(0);
    });

    it('extracts part-of edges as inverse of owns edges', async () => {
      const { extractCodeEntities } =
        await import('../extract-code-entities.mjs');
      const result = await extractCodeEntities({
        sourcePaths: [path.resolve('src/architecture/network/network.ts')],
      });

      const partOfEdges = result.edges.filter(
        (e) => e.relationship === 'part-of',
      );
      const ownsEdges = result.edges.filter((e) => e.relationship === 'owns');

      expect(partOfEdges.length).toBeGreaterThan(0);

      for (const partOfEdge of partOfEdges) {
        const correspondingOwnsEdge = ownsEdges.find(
          (owns) =>
            owns.source_qualified_name === partOfEdge.target_qualified_name &&
            owns.target_qualified_name === partOfEdge.source_qualified_name,
        );
        expect(correspondingOwnsEdge).toBeDefined();
      }
    });

    it('extracts depends-on edges with medium confidence', async () => {
      const { extractCodeEntities } =
        await import('../extract-code-entities.mjs');
      const result = await extractCodeEntities({
        sourcePaths: [path.resolve('src/architecture/network/network.ts')],
      });

      const dependsOnEdges = result.edges.filter(
        (e) => e.relationship === 'depends-on',
      );
      for (const edge of dependsOnEdges) {
        expect(edge.confidence).toBe('medium');
      }
    });

    it('returns entity maps for downstream use', async () => {
      const { extractCodeEntities } =
        await import('../extract-code-entities.mjs');
      const result = await extractCodeEntities({
        sourcePaths: [path.resolve('src/neat.ts')],
      });

      expect(result.moduleEntityMap).toBeInstanceOf(Map);
      expect(result.symbolEntityMap).toBeInstanceOf(Map);
    });

    it('limits depends-on edges per symbol to 20', async () => {
      const { extractCodeEntities } =
        await import('../extract-code-entities.mjs');
      const result = await extractCodeEntities({
        sourcePaths: [path.resolve('src/architecture/network/network.ts')],
      });

      const dependsOnCounts = new Map();
      for (const edge of result.edges.filter(
        (e) => e.relationship === 'depends-on',
      )) {
        const count =
          (dependsOnCounts.get(edge.source_qualified_name) ?? 0) + 1;
        dependsOnCounts.set(edge.source_qualified_name, count);
      }

      for (const [, count] of dependsOnCounts) {
        expect(count).toBeLessThanOrEqual(20);
      }
    });
  });
});
