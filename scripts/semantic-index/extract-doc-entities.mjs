/**
 * @module extract-doc-entities
 * @description Extract document entities (plan, skill, agent, demo, benchmark) from
 * markdown files by parsing headings and structured content. Creates entity records
 * and `contains` edges linking parent documents to their heading sections.
 *
 * Handles:
 * - `plans/` and `plans/completed/` → plan entities
 * - `.github/skills/` → skill entities
 * - `.github/agents/` → agent entities (with YAML frontmatter parsing)
 * - `examples/` → demo entities
 * - `benchmarks/` → benchmark entities
 *
 * @param {boolean} [--json] - Emit JSON output of extracted entities and edges.
 * @param {boolean} [--help] - Show help and exit.
 *
 * @returns {void} Exits 0 on success, 1 on error.
 */
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import {
  fail,
  parseCliArgs,
  printHelp,
  writeJsonOrText,
} from './cli-utils.mjs';
import { repoRoot } from './init-schema.mjs';

/**
 * Extract document entities from markdown files.
 *
 * For each document, produces:
 * 1. A parent entity (plan/skill/agent/demo/benchmark) with the correct qualified_name.
 * 2. `contains` edges from the parent entity to each heading section.
 *
 * @param {object} options - Extraction options.
 * @param {Array<{ filePath: string, family: string }>} options.documents - Document records to process.
 * @returns {Promise<{ entities: Array, edges: Array, entityMap: Map }>} Extracted entities and edges.
 */
export async function extractDocEntities(options) {
  const { documents } = options;
  const entities = [];
  const edges = [];
  const entityMap = new Map(); // qualified_name → entity

  for (const documentRecord of documents) {
    const { filePath, family } = documentRecord;
    const entityType = mapFamilyToEntityType(family);
    if (!entityType) continue;

    const absolutePath = path.join(repoRoot, filePath);
    let fileContent;
    try {
      fileContent = await readFile(absolutePath, 'utf8');
    } catch {
      // Skip files that cannot be read.
      continue;
    }

    const qualifiedName = constructDocQualifiedName(filePath, entityType);
    const extraMetadata = extractDocMetadata(fileContent, entityType, family);

    const parentEntity = {
      entity_type: entityType,
      name: extractDocName(qualifiedName, entityType),
      qualified_name: qualifiedName,
      file_path: filePath,
      module_path: null,
      signature_text: null,
      char_start: 0,
      char_end: fileContent.length,
      extra_metadata: JSON.stringify(extraMetadata),
    };

    entities.push(parentEntity);
    entityMap.set(qualifiedName, parentEntity);

    // Extract heading sections and create contains edges.
    const headings = extractHeadings(fileContent);
    for (const heading of headings) {
      const headingQualifiedName = `${qualifiedName}.${heading.slug}`;
      const headingEntity = {
        entity_type: entityType,
        name: heading.text,
        qualified_name: headingQualifiedName,
        file_path: filePath,
        module_path: null,
        signature_text: null,
        char_start: heading.charStart,
        char_end: heading.charEnd,
        extra_metadata: JSON.stringify({ heading_level: heading.level }),
      };

      entities.push(headingEntity);
      entityMap.set(headingQualifiedName, headingEntity);

      edges.push({
        source_qualified_name: qualifiedName,
        target_qualified_name: headingQualifiedName,
        relationship: 'contains',
        confidence: 'high',
      });
    }
  }

  return { entities, edges, entityMap };
}

/**
 * Map a document family to an entity type.
 *
 * @param {string} family - Document family from the corpus.
 * @returns {string | null} Entity type, or null if not a doc entity.
 */
export function mapFamilyToEntityType(family) {
  const familyMap = {
    plan: 'plan',
    'completed-plan': 'plan',
    skill: 'skill',
    agent: 'agent',
    demo: 'demo',
    benchmark: 'benchmark',
  };
  return familyMap[family] ?? null;
}

/**
 * Construct a qualified name for a document entity from its file path.
 *
 * | Entity type | `qualified_name` pattern | Example |
 * |---|---|---|
 * | plan | `plans/<stem>` | `plans/repo_cortex_advanced_rag` |
 * | skill | `skills/<skill-name>` | `skills/coverage-guard` |
 * | agent | `agents/<agent-name>` | `agents/04-implementing` |
 * | demo | `demos/<demo-name>` | `demos/flappy-bird-lstm` |
 * | benchmark | `benchmarks/<benchmark-name>` | `benchmarks/memory-optimization` |
 *
 * @param {string} filePath - Repository-relative file path.
 * @param {string} entityType - Entity type.
 * @returns {string} Qualified name.
 */
export function constructDocQualifiedName(filePath, entityType) {
  switch (entityType) {
    case 'plan': {
      // plans/Some_Plan.plans.md → plans/some_plan.plans
      // plans/Some_Plan.logs.md → plans/some_plan.logs
      // plans/completed/Some_Plan.md → plans/some_plan
      const relativePath = filePath.replace(/^plans\/completed\//, 'plans/');
      const stem = path
        .basename(relativePath, path.extname(relativePath))
        .replace(/\.md$/, '');
      return `plans/${kebabToSnake(stem)}`;
    }
    case 'skill': {
      // .github/skills/coverage-guard/SKILL.md → skills/coverage-guard
      const parts = filePath.split('/');
      const skillFolderIndex = parts.indexOf('skills');
      if (skillFolderIndex >= 0 && parts.length > skillFolderIndex + 1) {
        return `skills/${parts[skillFolderIndex + 1]}`;
      }
      return `skills/${path.basename(path.dirname(filePath))}`;
    }
    case 'agent': {
      // .github/agents/04-implementing.agent.md → agents/04-implementing
      const baseName = path.basename(filePath, '.md');
      return `agents/${baseName.replace(/\.agent$/, '')}`;
    }
    case 'demo': {
      // examples/flappy-bird-lstm/README.md → demos/flappy-bird-lstm
      const demoDir = path.dirname(filePath).replace(/^examples\//, '');
      return `demos/${demoDir || path.basename(filePath, '.md')}`;
    }
    case 'benchmark': {
      // benchmarks/memory-optimization/README.md → benchmarks/memory-optimization
      const benchDir = path.dirname(filePath).replace(/^benchmarks\//, '');
      return `benchmarks/${benchDir || path.basename(filePath, '.md')}`;
    }
    default:
      return filePath.replace(/\.md$/, '').replace(/\//g, '.');
  }
}

/**
 * Extract the display name from a qualified name for doc entities.
 *
 * @param {string} qualifiedName - Qualified name.
 * @param {string} entityType - Entity type.
 * @returns {string} Display name.
 */
function extractDocName(qualifiedName, entityType) {
  const prefix = `${entityType === 'completed-plan' ? 'plan' : entityType}/`;
  if (qualifiedName.startsWith(prefix)) {
    return qualifiedName.slice(prefix.length);
  }
  return qualifiedName;
}

/**
 * Extract metadata from a document file based on its entity type.
 *
 * For agents, parses YAML frontmatter to extract tier and skills.
 * For plans, extracts the status marker.
 *
 * @param {string} content - File content.
 * @param {string} entityType - Entity type.
 * @param {string} family - Document family.
 * @returns {object} Extracted metadata.
 */
export function extractDocMetadata(content, entityType, family) {
  const metadata = {};

  if (entityType === 'agent') {
    const frontmatter = parseYamlFrontmatter(content);
    if (frontmatter.tier) metadata.tier = frontmatter.tier;
    if (frontmatter.skills) metadata.skills = frontmatter.skills;
  }

  if (entityType === 'plan') {
    const statusMatch = content.match(/\*\*Status:\*\*\s*\[([^\]]+)\]/);
    if (statusMatch) metadata.status_marker = statusMatch[1];
  }

  return metadata;
}

/**
 * Parse YAML frontmatter from markdown content.
 *
 * Extracts key-value pairs from content delimited by `---` at the start.
 *
 * @param {string} content - Markdown content with optional YAML frontmatter.
 * @returns {object} Parsed key-value pairs.
 */
export function parseYamlFrontmatter(content) {
  const result = {};
  if (!content.startsWith('---')) return result;

  const endIndex = content.indexOf('---', 3);
  if (endIndex < 0) return result;

  const frontmatter = content.slice(3, endIndex).trim();
  for (const line of frontmatter.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;

    const colonIndex = trimmed.indexOf(':');
    if (colonIndex < 0) continue;

    const key = trimmed.slice(0, colonIndex).trim();
    let value = trimmed.slice(colonIndex + 1).trim();

    // Handle YAML arrays (e.g., skills: [...])
    if (value.startsWith('[') && value.endsWith(']')) {
      value = value
        .slice(1, -1)
        .split(',')
        .map((item) => item.trim().replace(/^['"]|['"]$/g, ''));
    } else if (value.startsWith("'") && value.endsWith("'")) {
      value = value.slice(1, -1);
    } else if (value.startsWith('"') && value.endsWith('"')) {
      value = value.slice(1, -1);
    }

    result[key] = value;
  }

  return result;
}

/**
 * Extract headings from markdown content.
 *
 * Parses all `#`, `##`, and `###` headings and returns their text,
 * slug, level, and character positions.
 *
 * @param {string} content - Markdown content.
 * @returns {Array<{ text: string, slug: string, level: number, charStart: number, charEnd: number }>} Headings.
 */
export function extractHeadings(content) {
  const headings = [];
  const lines = content.split('\n');
  let charOffset = 0;

  for (let lineIndex = 0; lineIndex < lines.length; lineIndex++) {
    const line = lines[lineIndex];
    const headingMatch = line.match(/^(#{1,3})\s+(.+)$/);
    if (!headingMatch) {
      charOffset += line.length + 1; // +1 for newline
      continue;
    }

    const level = headingMatch[1].length;
    const text = headingMatch[2].trim();
    const slug = slugifyHeading(text);
    const charStart = charOffset;
    // charEnd extends to the next heading or end of file.
    let charEnd = content.length;
    for (
      let nextLineIndex = lineIndex + 1;
      nextLineIndex < lines.length;
      nextLineIndex++
    ) {
      if (lines[nextLineIndex].match(/^#{1,3}\s+/)) {
        charEnd = charOffset + line.length;
        break;
      }
      charOffset +=
        lines[nextLineIndex - (nextLineIndex - lineIndex - 1)].length + 1;
    }

    headings.push({
      text,
      slug,
      level,
      charStart,
      charEnd,
    });

    charOffset += line.length + 1;
  }

  return headings;
}

/**
 * Convert a heading text to a URL-safe slug.
 *
 * @param {string} text - Heading text.
 * @returns {string} Slugified heading text.
 */
function slugifyHeading(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

/**
 * Convert PascalCase or CamelCase to snake_case.
 *
 * @param {string} value - Input string.
 * @returns {string} snake_case string.
 */
function kebabToSnake(value) {
  return value
    .replace(/([a-z])([A-Z])/g, '$1_$2')
    .replace(/-/g, '_')
    .toLowerCase();
}

/**
 * CLI entrypoint for doc entity extraction.
 *
 * @returns {Promise<void>}
 */
async function main() {
  const args = parseCliArgs(process.argv.slice(2));
  if (args.help) {
    printHelp({
      title: 'Doc Entity Extractor',
      usage: 'node scripts/semantic-index/extract-doc-entities.mjs [--json]',
      options: [
        '--json   Emit JSON output of extracted entities and edges.',
        '--help   Show this help.',
      ],
    });
    return;
  }

  try {
    // For CLI usage, scan default doc directories.
    const fg = (await import('fast-glob')).default;
    const docSources = [
      {
        family: 'plan',
        patterns: ['plans/**/*.md'],
        ignore: ['plans/completed/**'],
      },
      { family: 'completed-plan', patterns: ['plans/completed/**/*.md'] },
      { family: 'skill', patterns: ['.github/skills/**/SKILL.md'] },
      { family: 'agent', patterns: ['.github/agents/*.agent.md'] },
      { family: 'demo', patterns: ['examples/**/README.md'] },
      { family: 'benchmark', patterns: ['benchmarks/README.md'] },
    ];

    const documents = [];
    for (const source of docSources) {
      const entries = await fg(source.patterns, {
        cwd: repoRoot,
        absolute: false,
        onlyFiles: true,
        dot: true,
        ignore: source.ignore ?? [],
      });
      for (const filePath of entries) {
        documents.push({ filePath, family: source.family });
      }
    }

    const result = await extractDocEntities({ documents });
    writeJsonOrText(result, Boolean(args.json), (payload) => {
      const entityCounts = {};
      for (const entity of payload.entities) {
        entityCounts[entity.entity_type] =
          (entityCounts[entity.entity_type] ?? 0) + 1;
      }
      return `Doc entities: ${payload.entities.length} (${Object.entries(
        entityCounts,
      )
        .map(([k, v]) => `${k}=${v}`)
        .join(', ')}), edges: ${payload.edges.length}`;
    });
  } catch (error) {
    fail(
      error instanceof Error ? error.message : String(error),
      Boolean(args.json),
    );
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href)
  await main();
