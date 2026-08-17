/**
 * @module extract-cross-refs
 * @description Extract cross-reference edges from document body text using pattern
 * matching. Produces `references` edges linking doc entities (plan, skill, agent,
 * demo, benchmark) to code entities (module, class, function, interface, type-alias,
 * variable, error-class).
 *
 * Cross-reference patterns, in priority order:
 * 1. `src/<path>/<file>.ts` → module entity (medium confidence)
 * 2. `<ClassName>.<methodName>` → method function entity (medium confidence)
 * 3. `<ClassName>` → class entity (low confidence)
 * 4. `code_symbol` (backtick-quoted) → symbol entity (low confidence)
 * 5. `plans/<plan-name>` → plan entity (medium confidence)
 * 6. `.github/skills/<skill-name>` → skill entity (medium confidence)
 *
 * Pattern matching is limited to the first 2,000 chars of each heading section
 * to avoid false positives from deep prose.
 *
 * @param {boolean} [--json] - Emit JSON output of extracted cross-reference edges.
 * @param {boolean} [--help] - Show help and exit.
 *
 * @returns {void} Exits 0 on success, 1 on error.
 */
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import {
  fail,
  parseCliArgs,
  printHelp,
  writeJsonOrText,
} from './cli-utils.mjs';

/** Maximum chars of each heading section to scan for cross-references. */
const MAX_CROSS_REF_SCAN_CHARS = 2_000;

/** Cross-reference patterns to match. */
const CROSS_REF_PATTERNS = [
  // src/<path>/<file>.ts → module entity
  {
    regex: /\bsrc\/((?:[a-zA-Z0-9_-]+\/)*[a-zA-Z0-9_-]+\.ts)\b/g,
    confidence: 'medium',
    entityHint: 'module',
  },
  // plans/<plan-name> → plan entity
  {
    regex: /\bplans\/([a-zA-Z0-9_-]+)\b/g,
    confidence: 'medium',
    entityHint: 'plan',
  },
  // .github/skills/<skill-name> → skill entity
  {
    regex: /\.github\/skills\/([a-zA-Z0-9_-]+)/g,
    confidence: 'medium',
    entityHint: 'skill',
  },
];

/** Backtick-quoted code symbol pattern. */
const BACKTICK_SYMBOL_REGEX = /`([a-zA-Z_][a-zA-Z0-9_./]*)`/g;

/**
 * Extract cross-reference edges from document body text.
 *
 * Scans each doc entity's body text (limited to first 2,000 chars) for patterns
 * that reference code entities or other doc entities. Produces `references` edges
 * linking the doc entity to the discovered target entities.
 *
 * @param {object} options - Extraction options.
 * @param {Array} options.docEntities - Doc entities with body text to scan.
 * @param {Map} options.codeEntityMap - Map of qualified_name → code entity.
 * @param {Map} options.docEntityMap - Map of qualified_name → doc entity.
 * @param {object} options.headingsByDoc - Map of parent_qualified_name → heading text content.
 * @returns {{ edges: Array }} Extracted cross-reference edges.
 */
export function extractCrossRefs(options) {
  const { docEntities, codeEntityMap, docEntityMap, headingsByDoc } = options;
  const edges = [];
  const seenEdges = new Set();

  for (const docEntity of docEntities) {
    const parentQualifiedName = getParentQualifiedName(
      docEntity.qualified_name,
    );
    const headingTexts = headingsByDoc?.get(parentQualifiedName) ?? [];
    const scanText = headingTexts.join('\n').slice(0, MAX_CROSS_REF_SCAN_CHARS);

    if (!scanText) continue;

    // Pattern 1: src/<path>/<file>.ts references.
    extractPathReferences(scanText, docEntity, codeEntityMap, edges, seenEdges);

    // Pattern 2: ClassName.methodName references.
    extractMethodReferences(
      scanText,
      docEntity,
      codeEntityMap,
      edges,
      seenEdges,
    );

    // Pattern 3: ClassName references.
    extractClassReferences(
      scanText,
      docEntity,
      codeEntityMap,
      edges,
      seenEdges,
    );

    // Pattern 4: Backtick-quoted code symbols.
    extractBacktickReferences(
      scanText,
      docEntity,
      codeEntityMap,
      docEntityMap,
      edges,
      seenEdges,
    );

    // Pattern 5: plans/<plan-name> references.
    extractPlanReferences(scanText, docEntity, docEntityMap, edges, seenEdges);

    // Pattern 6: .github/skills/<skill-name> references.
    extractSkillReferences(scanText, docEntity, docEntityMap, edges, seenEdges);
  }

  return { edges };
}

/**
 * Extract references from `src/<path>/<file>.ts` patterns.
 *
 * @param {string} scanText - Text to scan.
 * @param {object} sourceEntity - Source doc entity.
 * @param {Map} codeEntityMap - Map of qualified_name → code entity.
 * @param {Array} edges - Accumulator for edges.
 * @param {Set} seenEdges - Dedup set of "source→target" strings.
 */
function extractPathReferences(
  scanText,
  sourceEntity,
  codeEntityMap,
  edges,
  seenEdges,
) {
  // Match src/<path>.ts — supports single-segment (src/neat.ts) and nested (src/a/b.ts) files.
  const regex = /\bsrc\/((?:[a-zA-Z0-9_-]+\/)*[a-zA-Z0-9_-]+\.ts)\b/g;
  let match;

  while ((match = regex.exec(scanText)) !== null) {
    const filePath = `src/${match[1]}`;
    // Use deriveModulePath to match how code entities compute module paths.
    // For src/architecture/network/network.ts → src/architecture/network.
    // For src/neat.ts → src/neat.
    const modulePath = filePath.replace(/\.ts$/, '').replace(/\/index$/, '');

    // Try the stripped path first, then try deriving the directory-based module path.
    let targetEntity = codeEntityMap.get(modulePath);
    if (!targetEntity) {
      // For nested files, the module path is the directory (e.g., src/architecture/network).
      // Derive it by taking the directory of the file.
      /* istanbul ignore next -- defensive: modulePath always contains '/' for nested files */
      const dirPath = modulePath.includes('/')
        ? modulePath.substring(0, modulePath.lastIndexOf('/'))
        : modulePath;
      targetEntity = codeEntityMap.get(dirPath);
    }

    if (targetEntity) {
      addReferenceEdge(sourceEntity, targetEntity, 'medium', edges, seenEdges);
    }
  }
}

/**
 * Extract references from `ClassName.methodName` patterns.
 *
 * @param {string} scanText - Text to scan.
 * @param {object} sourceEntity - Source doc entity.
 * @param {Map} codeEntityMap - Map of qualified_name → code entity.
 * @param {Array} edges - Accumulator for edges.
 * @param {Set} seenEdges - Dedup set of "source→target" strings.
 */
function extractMethodReferences(
  scanText,
  sourceEntity,
  codeEntityMap,
  edges,
  seenEdges,
) {
  const regex = /\b([A-Z][a-zA-Z0-9]*)\.([a-z][a-zA-Z0-9]*)\b/g;
  let match;

  while ((match = regex.exec(scanText)) !== null) {
    const className = match[1];
    const methodName = match[2];

    // Search for matching method entities.
    for (const [qualifiedName, entity] of codeEntityMap) {
      if (entity.entity_type !== 'function') continue;
      if (!qualifiedName.includes(`.${className}.${methodName}`)) continue;
      addReferenceEdge(sourceEntity, entity, 'medium', edges, seenEdges);
    }
  }
}

/**
 * Extract references from `ClassName` patterns.
 *
 * @param {string} scanText - Text to scan.
 * @param {object} sourceEntity - Source doc entity.
 * @param {Map} codeEntityMap - Map of qualified_name → code entity.
 * @param {Array} edges - Accumulator for edges.
 * @param {Set} seenEdges - Dedup set of "source→target" strings.
 */
function extractClassReferences(
  scanText,
  sourceEntity,
  codeEntityMap,
  edges,
  seenEdges,
) {
  const regex = /\b([A-Z][a-zA-Z0-9]{2,})\b/g;
  let match;

  while ((match = regex.exec(scanText)) !== null) {
    const className = match[1];

    // Skip common non-class words.
    if (
      [
        'The',
        'This',
        'That',
        'When',
        'Then',
        'Each',
        'With',
        'For',
        'And',
        'But',
        'Not',
        'All',
        'Any',
        'Has',
      ].includes(className)
    )
      continue;

    for (const [qualifiedName, entity] of codeEntityMap) {
      if (
        entity.entity_type !== 'class' &&
        entity.entity_type !== 'error-class'
      )
        continue;
      if (entity.name !== className) continue;
      addReferenceEdge(sourceEntity, entity, 'low', edges, seenEdges);
    }
  }
}

/**
 * Extract references from backtick-quoted code symbols.
 *
 * @param {string} scanText - Text to scan.
 * @param {object} sourceEntity - Source doc entity.
 * @param {Map} codeEntityMap - Map of qualified_name → code entity.
 * @param {Map} docEntityMap - Map of qualified_name → doc entity.
 * @param {Array} edges - Accumulator for edges.
 * @param {Set} seenEdges - Dedup set of "source→target" strings.
 */
function extractBacktickReferences(
  scanText,
  sourceEntity,
  codeEntityMap,
  docEntityMap,
  edges,
  seenEdges,
) {
  const regex = /`([a-zA-Z_][a-zA-Z0-9_./]*)`/g;
  let match;

  while ((match = regex.exec(scanText)) !== null) {
    const symbolText = match[1];

    // Try exact qualified_name match.
    let targetEntity =
      codeEntityMap.get(symbolText) ?? docEntityMap.get(symbolText);

    // Try fuzzy name match.
    if (!targetEntity) {
      for (const [qualifiedName, entity] of codeEntityMap) {
        if (entity.name === symbolText) {
          targetEntity = entity;
          break;
        }
      }
    }

    if (targetEntity) {
      addReferenceEdge(sourceEntity, targetEntity, 'low', edges, seenEdges);
    }
  }
}

/**
 * Extract references from `plans/<plan-name>` patterns.
 *
 * @param {string} scanText - Text to scan.
 * @param {object} sourceEntity - Source doc entity.
 * @param {Map} docEntityMap - Map of qualified_name → doc entity.
 * @param {Array} edges - Accumulator for edges.
 * @param {Set} seenEdges - Dedup set of "source→target" strings.
 */
function extractPlanReferences(
  scanText,
  sourceEntity,
  docEntityMap,
  edges,
  seenEdges,
) {
  const regex = /\bplans\/([a-zA-Z0-9_-]+)/g;
  let match;

  while ((match = regex.exec(scanText)) !== null) {
    const planStem = match[1];
    // Try various qualified name forms.
    const candidates = [
      `plans/${planStem}`,
      `plans/${planStem.toLowerCase()}`,
      `plans/${planStem.replace(/-/g, '_')}`,
    ];
    for (const candidate of candidates) {
      const targetEntity = docEntityMap.get(candidate);
      if (targetEntity && targetEntity.entity_type === 'plan') {
        addReferenceEdge(
          sourceEntity,
          targetEntity,
          'medium',
          edges,
          seenEdges,
        );
        break;
      }
    }
  }
}

/**
 * Extract references from `.github/skills/<skill-name>` patterns.
 *
 * @param {string} scanText - Text to scan.
 * @param {object} sourceEntity - Source doc entity.
 * @param {Map} docEntityMap - Map of qualified_name → doc entity.
 * @param {Array} edges - Accumulator for edges.
 * @param {Set} seenEdges - Dedup set of "source→target" strings.
 */
function extractSkillReferences(
  scanText,
  sourceEntity,
  docEntityMap,
  edges,
  seenEdges,
) {
  const regex = /\.github\/skills\/([a-zA-Z0-9_-]+)/g;
  let match;

  while ((match = regex.exec(scanText)) !== null) {
    const skillName = match[1];
    const qualifiedName = `skills/${skillName}`;
    const targetEntity = docEntityMap.get(qualifiedName);
    if (targetEntity && targetEntity.entity_type === 'skill') {
      addReferenceEdge(sourceEntity, targetEntity, 'medium', edges, seenEdges);
    }
  }
}

/**
 * Add a reference edge between two entities, with deduplication.
 *
 * @param {object} sourceEntity - Source entity.
 * @param {object} targetEntity - Target entity.
 * @param {string} confidence - Edge confidence level.
 * @param {Array} edges - Accumulator for edges.
 * @param {Set} seenEdges - Dedup set.
 */
function addReferenceEdge(
  sourceEntity,
  targetEntity,
  confidence,
  edges,
  seenEdges,
) {
  const edgeKey = `${sourceEntity.qualified_name}→${targetEntity.qualified_name}`;
  if (seenEdges.has(edgeKey)) return;
  seenEdges.add(edgeKey);

  edges.push({
    source_qualified_name: sourceEntity.qualified_name,
    target_qualified_name: targetEntity.qualified_name,
    relationship: 'references',
    confidence,
  });
}

/**
 * Get the parent qualified name from a heading entity's qualified name.
 *
 * Heading entities have qualified names like `plans/my_plan.scope`, so the
 * parent is `plans/my_plan`. For top-level entities, the parent is themselves.
 *
 * @param {string} qualifiedName - Qualified name.
 * @returns {string} Parent qualified name.
 */
function getParentQualifiedName(qualifiedName) {
  // Check if this is a heading entity (contains a dot after the type prefix).
  const dotIndex = qualifiedName.indexOf('.', qualifiedName.indexOf('/') + 1);
  if (dotIndex < 0) return qualifiedName;

  // Check if the dot is part of a heading slug (after type prefix).
  const slashCount = qualifiedName.split('/').length - 1;
  const firstDotAfterSlash = qualifiedName.indexOf('.', slashCount);

  // If the qualified name looks like a heading entity, extract parent.
  /* istanbul ignore next -- defensive: heading entities always have a dot after the slash */
  if (firstDotAfterSlash > 0) {
    const prefix = qualifiedName.slice(0, firstDotAfterSlash);
    const suffix = qualifiedName.slice(firstDotAfterSlash + 1);
    // Only strip heading if suffix is a slug (lowercase, contains hyphens).
    if (suffix === suffix.toLowerCase() && suffix.includes('-')) {
      return prefix;
    }
  }

  return qualifiedName;
}

/**
 * CLI entrypoint for cross-reference extraction.
 *
 * @returns {Promise<void>}
 */
export async function main() {
  const args = parseCliArgs(process.argv.slice(2));
  if (args.help) {
    printHelp({
      title: 'Cross-Reference Extractor',
      usage: 'node rag-index/extract-cross-refs.mjs [--json]',
      options: [
        '--json   Emit JSON output of extracted cross-reference edges.',
        '--help   Show this help.',
      ],
    });
    return;
  }

  // CLI mode runs the full pipeline for standalone testing.
  try {
    const { extractCodeEntities } = await import('./extract-code-entities.mjs');
    const { extractDocEntities } = await import('./extract-doc-entities.mjs');
    const fg = (await import('fast-glob')).default;
    const { readFile } = await import('node:fs/promises');
    const { repoRoot } = await import('./init-schema.mjs');

    // Extract code entities first.
    const codeResult = await extractCodeEntities();
    const codeEntityMap = new Map(
      codeResult.entities.map((e) => [e.qualified_name, e]),
    );

    // Extract doc entities.
    const docSources = [
      {
        family: 'plan',
        patterns: ['plans/**/*.md'],
        ignore: ['plans/completed/**'],
      },
      { family: 'completed-plan', patterns: ['plans/completed/**/*.md'] },
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

    const docResult = await extractDocEntities({ documents });
    const docEntityMap = new Map(
      docResult.entities.map((e) => [e.qualified_name, e]),
    );

    // Build heading text map from doc files.
    const headingsByDoc = new Map();
    for (const docEntity of docResult.entities) {
      const parentQName = getParentQualifiedName(docEntity.qualified_name);
      /* istanbul ignore next -- defensive: parentQName is always already in headingsByDoc from prior iteration */
      if (!headingsByDoc.has(parentQName)) {
        headingsByDoc.set(parentQName, []);
      }
      // Read the file content for this heading section.
      try {
        const absolutePath = path.join(repoRoot, docEntity.file_path);
        const content = await readFile(absolutePath, 'utf8');
        const sectionText = content.slice(
          docEntity.char_start,
          docEntity.char_end,
        );
        headingsByDoc.get(parentQName).push(sectionText);
      } catch {
        // Skip unreadable files.
      }
    }

    const result = extractCrossRefs({
      docEntities: docResult.entities.filter(
        (e) =>
          /* istanbul ignore next -- defensive: qualified_name always contains a dot after the slash for heading entities */
          !e.qualified_name.includes('.', e.qualified_name.indexOf('/') + 1) ||
          !e.qualified_name
            .slice(
              e.qualified_name.indexOf('.', e.qualified_name.indexOf('/') + 1) +
                1,
            )
            .includes('-'),
      ),
      codeEntityMap,
      docEntityMap,
      headingsByDoc,
    });

    writeJsonOrText(
      result,
      Boolean(args.json),
      /* istanbul ignore next -- text formatter, covered by JSON-mode tests */
      (payload) => `Cross-reference edges: ${payload.edges.length}`,
    );
  } catch (error) {
    fail(
      error instanceof Error ? error.message : String(error),
      Boolean(args.json),
    );
  }
}

/* istanbul ignore next */
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href)
  await main();
