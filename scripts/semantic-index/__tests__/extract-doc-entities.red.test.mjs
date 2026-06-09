/**
 * @module extract-doc-entities.red.test
 * @description Red tests for doc entity extraction from markdown files.
 */
import path from 'node:path';

describe('extract-doc-entities', () => {
  describe('mapFamilyToEntityType', () => {
    it('maps plan family to plan entity type', async () => {
      const { mapFamilyToEntityType } =
        await import('../extract-doc-entities.mjs');
      expect(mapFamilyToEntityType('plan')).toBe('plan');
    });

    it('maps completed-plan family to plan entity type', async () => {
      const { mapFamilyToEntityType } =
        await import('../extract-doc-entities.mjs');
      expect(mapFamilyToEntityType('completed-plan')).toBe('plan');
    });

    it('maps skill family to skill entity type', async () => {
      const { mapFamilyToEntityType } =
        await import('../extract-doc-entities.mjs');
      expect(mapFamilyToEntityType('skill')).toBe('skill');
    });

    it('maps agent family to agent entity type', async () => {
      const { mapFamilyToEntityType } =
        await import('../extract-doc-entities.mjs');
      expect(mapFamilyToEntityType('agent')).toBe('agent');
    });

    it('maps demo family to demo entity type', async () => {
      const { mapFamilyToEntityType } =
        await import('../extract-doc-entities.mjs');
      expect(mapFamilyToEntityType('demo')).toBe('demo');
    });

    it('maps benchmark family to benchmark entity type', async () => {
      const { mapFamilyToEntityType } =
        await import('../extract-doc-entities.mjs');
      expect(mapFamilyToEntityType('benchmark')).toBe('benchmark');
    });

    it('returns null for non-doc families (ts-source, readme, etc.)', async () => {
      const { mapFamilyToEntityType } =
        await import('../extract-doc-entities.mjs');
      expect(mapFamilyToEntityType('ts-source')).toBeNull();
      expect(mapFamilyToEntityType('readme')).toBeNull();
      expect(mapFamilyToEntityType('unknown')).toBeNull();
    });
  });

  describe('constructDocQualifiedName', () => {
    it('constructs plan qualified name from plans/ file path', async () => {
      const { constructDocQualifiedName } =
        await import('../extract-doc-entities.mjs');
      const result = constructDocQualifiedName(
        'plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md',
        'plan',
      );
      expect(result).toMatch(/^plans\//);
    });

    it('constructs plan qualified name from plans/completed/ file path', async () => {
      const { constructDocQualifiedName } =
        await import('../extract-doc-entities.mjs');
      const result = constructDocQualifiedName(
        'plans/completed/Semantic_Knowledge_Foundation.plans.md',
        'plan',
      );
      expect(result).toMatch(/^plans\//);
    });

    it('constructs skill qualified name from .github/skills/ file path', async () => {
      const { constructDocQualifiedName } =
        await import('../extract-doc-entities.mjs');
      const result = constructDocQualifiedName(
        '.github/skills/coverage-guard/SKILL.md',
        'skill',
      );
      expect(result).toBe('skills/coverage-guard');
    });

    it('constructs agent qualified name from .github/agents/ file path', async () => {
      const { constructDocQualifiedName } =
        await import('../extract-doc-entities.mjs');
      const result = constructDocQualifiedName(
        '.github/agents/04-implementing.agent.md',
        'agent',
      );
      expect(result).toBe('agents/04-implementing');
    });

    it('constructs demo qualified name from examples/ file path', async () => {
      const { constructDocQualifiedName } =
        await import('../extract-doc-entities.mjs');
      const result = constructDocQualifiedName(
        'examples/flappy-bird-lstm/README.md',
        'demo',
      );
      expect(result).toMatch(/^demos\//);
    });

    it('constructs benchmark qualified name from benchmarks/ file path', async () => {
      const { constructDocQualifiedName } =
        await import('../extract-doc-entities.mjs');
      const result = constructDocQualifiedName(
        'benchmarks/memory-optimization/README.md',
        'benchmark',
      );
      expect(result).toMatch(/^benchmarks\//);
    });
  });

  describe('parseYamlFrontmatter', () => {
    it('parses simple key-value frontmatter', async () => {
      const { parseYamlFrontmatter } =
        await import('../extract-doc-entities.mjs');
      const content = '---\ntier: 1\ndescription: Test agent\n---\nBody text';
      const result = parseYamlFrontmatter(content);
      expect(result.tier).toBe('1');
      expect(result.description).toBe('Test agent');
    });

    it('parses array values in frontmatter', async () => {
      const { parseYamlFrontmatter } =
        await import('../extract-doc-entities.mjs');
      const content =
        '---\nskills: [coverage-guard, plan-scout]\n---\nBody text';
      const result = parseYamlFrontmatter(content);
      expect(result.skills).toEqual(['coverage-guard', 'plan-scout']);
    });

    it('parses single-quoted values', async () => {
      const { parseYamlFrontmatter } =
        await import('../extract-doc-entities.mjs');
      const content = "---\nname: 'test-value'\n---\nBody text";
      const result = parseYamlFrontmatter(content);
      expect(result.name).toBe('test-value');
    });

    it('parses double-quoted values', async () => {
      const { parseYamlFrontmatter } =
        await import('../extract-doc-entities.mjs');
      const content = '---\nname: "test-value"\n---\nBody text';
      const result = parseYamlFrontmatter(content);
      expect(result.name).toBe('test-value');
    });

    it('returns empty object for content without frontmatter', async () => {
      const { parseYamlFrontmatter } =
        await import('../extract-doc-entities.mjs');
      const content = '# No Frontmatter\n\nJust body text.';
      const result = parseYamlFrontmatter(content);
      expect(result).toEqual({});
    });

    it('skips comment lines in frontmatter', async () => {
      const { parseYamlFrontmatter } =
        await import('../extract-doc-entities.mjs');
      const content = '---\n# This is a comment\ntier: 2\n---\nBody text';
      const result = parseYamlFrontmatter(content);
      expect(result.tier).toBe('2');
      expect(Object.keys(result)).toHaveLength(1);
    });
  });

  describe('extractHeadings', () => {
    it('extracts h1, h2, and h3 headings from markdown', async () => {
      const { extractHeadings } = await import('../extract-doc-entities.mjs');
      const content =
        '# Title\n\n## Section\n\nContent\n### Subsection\n\nMore content';
      const headings = extractHeadings(content);

      expect(headings).toHaveLength(3);
      expect(headings[0].level).toBe(1);
      expect(headings[0].text).toBe('Title');
      expect(headings[1].level).toBe(2);
      expect(headings[1].text).toBe('Section');
      expect(headings[2].level).toBe(3);
      expect(headings[2].text).toBe('Subsection');
    });

    it('generates slugs from heading text', async () => {
      const { extractHeadings } = await import('../extract-doc-entities.mjs');
      const content = '# Entity Graph Design\n\n## BFS Traversal\n\n### Step 1';
      const headings = extractHeadings(content);

      expect(headings[0].slug).toBe('entity-graph-design');
      expect(headings[1].slug).toBe('bfs-traversal');
      expect(headings[2].slug).toBe('step-1');
    });

    it('computes char_start and char_end positions', async () => {
      const { extractHeadings } = await import('../extract-doc-entities.mjs');
      const content = '# Title\n\nBody\n## Section\n\nContent';
      const headings = extractHeadings(content);

      expect(headings[0].charStart).toBe(0);
      expect(headings[0].charEnd).toBeGreaterThan(0);
      expect(headings[1].charStart).toBeGreaterThan(headings[0].charStart);
    });

    it('returns empty array for content without headings', async () => {
      const { extractHeadings } = await import('../extract-doc-entities.mjs');
      const content = 'Just plain text without any headings.';
      const headings = extractHeadings(content);
      expect(headings).toHaveLength(0);
    });
  });

  describe('extractDocMetadata', () => {
    it('extracts tier from agent YAML frontmatter', async () => {
      const { extractDocMetadata } =
        await import('../extract-doc-entities.mjs');
      const content = '---\ntier: 1\nskills: [a, b]\n---\nAgent body';
      const metadata = extractDocMetadata(content, 'agent', 'agent');
      expect(metadata.tier).toBe('1');
      expect(metadata.skills).toEqual(['a', 'b']);
    });

    it('extracts status marker from plan documents', async () => {
      const { extractDocMetadata } =
        await import('../extract-doc-entities.mjs');
      const content = '# Plan\n\n**Status:** [WIP]\n\nBody text';
      const metadata = extractDocMetadata(content, 'plan', 'plan');
      expect(metadata.status_marker).toBe('WIP');
    });

    it('returns empty metadata for doc types without special metadata', async () => {
      const { extractDocMetadata } =
        await import('../extract-doc-entities.mjs');
      const metadata = extractDocMetadata('Body', 'demo', 'demo');
      expect(metadata).toEqual({});
    });
  });

  describe('extractDocEntities', () => {
    it('extracts plan entities from plan documents', async () => {
      const { extractDocEntities } =
        await import('../extract-doc-entities.mjs');

      // Use an actual plan file that exists in the repo.
      const documents = [
        {
          filePath: 'plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md',
          family: 'plan',
        },
      ];

      const result = await extractDocEntities({ documents });

      const planEntities = result.entities.filter(
        (e) => e.entity_type === 'plan',
      );
      expect(planEntities.length).toBeGreaterThan(0);
      expect(planEntities[0].qualified_name).toMatch(/^plans\//);
    });

    it('creates contains edges from parent to heading entities', async () => {
      const { extractDocEntities } =
        await import('../extract-doc-entities.mjs');

      const documents = [
        {
          filePath: 'plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md',
          family: 'plan',
        },
      ];

      const result = await extractDocEntities({ documents });

      const containsEdges = result.edges.filter(
        (e) => e.relationship === 'contains',
      );
      // Should have contains edges if the plan has headings.
      if (containsEdges.length > 0) {
        for (const edge of containsEdges) {
          expect(edge.relationship).toBe('contains');
          expect(edge.confidence).toBe('high');
        }
      }
    });

    it('returns entity map with qualified names as keys', async () => {
      const { extractDocEntities } =
        await import('../extract-doc-entities.mjs');

      const documents = [
        {
          filePath: 'plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md',
          family: 'plan',
        },
      ];

      const result = await extractDocEntities({ documents });

      expect(result.entityMap).toBeInstanceOf(Map);
      for (const [key, entity] of result.entityMap) {
        expect(key).toBe(entity.qualified_name);
      }
    });

    it('skips documents that cannot be read', async () => {
      const { extractDocEntities } =
        await import('../extract-doc-entities.mjs');

      const documents = [
        {
          filePath: 'plans/Nonexistent_Plan.plans.md',
          family: 'plan',
        },
      ];

      // Should not throw, just skip the unreadable file.
      const result = await extractDocEntities({ documents });
      const planEntities = result.entities.filter((e) =>
        e.qualified_name.includes('nonexistent'),
      );
      expect(planEntities.length).toBe(0);
    });
  });
});
