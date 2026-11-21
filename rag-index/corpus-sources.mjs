/**
 * @module corpus-sources
 * @description The published corpus-family contract for the NeatapticTS semantic index.
 *
 * This module is the single owner of the corpus contract:
 *
 * - the published family set (the nine corpus families),
 * - the global ignore list,
 * - the docs-generator target roots, PUBLISHED AS PLAIN DATA,
 * - the generated-README exclusion, derived from those target roots,
 * - the handwritten-README allow-list (the example README paths the generator does not own),
 * - the generated session-snapshot exclusion, and
 * - the test-family membership.
 *
 * `rag-index/build-index.mjs` and `rag-index/validate-index.mjs` import this module instead of
 * declaring their own copies of the corpus rules, so the builder and the validator can never
 * disagree about which documents belong to the corpus.
 *
 * The module is PLAIN ESM + PLAIN DATA: it loads no TypeScript, reads no file at import time and
 * therefore costs every importer nothing beyond the data it publishes. The link to the
 * docs-generator source of truth is a CONTRACT OBLIGATION rather than a runtime parse: the
 * published table MUST mirror the `DOCS_TARGETS` constant of
 * `scripts/generate-docs/generate-docs.constants.ts`. The automated checker for that obligation is
 * the designated guard of criterion `AC-202-B` (the parity home
 * `rag-index/__tests__/corpus-parity.red.test.mjs`, owned by slice `P1-S2-1`). Until that guard
 * home exists, the obligation is MANUAL and the published table CAN drift from the generator
 * silently, so any change to the generator constants must be mirrored here by hand.
 */
/**
 * The docs-generator target roots, PUBLISHED AS PLAIN DATA.
 *
 * Each entry names one source tree the folder README generator owns, so every README below that
 * root is a generated document rather than a handwritten one.
 *
 * The table MIRRORS the `DOCS_TARGETS` constant of `scripts/generate-docs/generate-docs.constants.ts`
 * - same keys, same declaration order, same source roots and the same `excludeRootSourceFiles`
 * flags - and every root is a repo-relative POSIX path, so the published contract does not depend
 * on the working directory.
 *
 * The table is published rather than parsed so that importing this module costs no TypeScript
 * parse: the generator-to-contract link belongs to the designated guard of criterion `AC-202-B`
 * (the parity home `rag-index/__tests__/corpus-parity.red.test.mjs`, owned by slice `P1-S2-1`)
 * instead of to a runtime read of the generator constants. Until that guard home exists, mirroring
 * `DOCS_TARGETS` here is a MANUAL obligation.
 *
 * @type {ReadonlyArray<{ name: string, root: string, excludeRootSourceFiles: boolean }>}
 */
export const DOCS_TARGET_ROOTS = [
  { name: 'src', root: 'src', excludeRootSourceFiles: false },
  {
    name: 'asciiMaze',
    root: 'examples/asciiMaze',
    excludeRootSourceFiles: true,
  },
  {
    name: 'flappy-bird',
    root: 'examples/flappy_bird',
    excludeRootSourceFiles: true,
  },
  {
    name: 'racing-curriculum',
    root: 'examples/racing_curriculum',
    excludeRootSourceFiles: true,
  },
  {
    name: 'shared-network-visualization',
    root: 'examples/shared/network-visualization',
    excludeRootSourceFiles: false,
  },
  {
    name: 'neatRacing',
    root: 'examples/neatRacing',
    excludeRootSourceFiles: true,
  },
];

/**
 * The closed allow-list of handwritten example READMEs.
 *
 * These README files are authored by hand and stay in the corpus even when they sit below a
 * docs-generator target root; every other README below a target root is generated output.
 *
 * @type {ReadonlyArray<string>}
 */
export const HANDWRITTEN_EXAMPLE_READMES = [
  'examples/README.md',
  'examples/asciiMaze/README.md',
  'examples/evolveXor/README.md',
  'examples/flappy_bird/README.md',
  'examples/helloNetwork/README.md',
  'examples/neatChat/README.md',
  'examples/neatChat/memory/README.md',
  'examples/neatRacing/README.md',
  'examples/neatenstein/README.md',
  'examples/neatenstein/browser-entry/README.md',
  'examples/racing_curriculum/README.md',
  'examples/sequenceReset/README.md',
  'examples/shared/semantic/README.md',
];

/**
 * The generated session snapshot that must leave the corpus.
 *
 * The snapshot is a runtime artifact of one example session, not authored source.
 *
 * @type {string}
 */
export const SESSION_SNAPSHOT_PATH =
  'examples/neatChat/default-pretrained-session-snapshot.ts';

/**
 * The generated-README exclusion, derived from the docs-generator target roots.
 *
 * A docs-generator target root owns every README below it. The root README of a target stays in
 * the corpus only when the handwritten allow-list claims it, so the exclusion for such a root
 * starts one directory below the root.
 *
 * @type {ReadonlyArray<string>}
 */
export const GENERATED_README_IGNORE = DOCS_TARGET_ROOTS.map(({ root }) =>
  HANDWRITTEN_EXAMPLE_READMES.includes(`${root}/README.md`)
    ? `${root}/*/**/README.md`
    : `${root}/**/README.md`,
);

/**
 * Paths that never enter the corpus, wherever they are matched from.
 *
 * @type {ReadonlyArray<string>}
 */
export const GLOBAL_IGNORE = [
  '**/node_modules/**',
  '**/dist/**',
  '**/build/**',
  '**/out/**',
  '**/coverage/**',
  '**/.cache/**',
  '**/artifacts/**',
  'rag-index/data/**',
  'rag-index/models/**',
  'rag-index/snapshots/**',
  'rag-index/freshness-proofs/**',
];

/**
 * The published corpus families.
 *
 * Every family names the glob patterns that select its documents and the ignore rules that keep
 * the published exclusions out of it. The families are disjoint, so a document belongs to exactly
 * one family.
 *
 * @type {ReadonlyArray<{ family: string, patterns: string[], ignore?: string[] }>}
 */
export const CORPUS_SOURCES = [
  {
    family: 'ts-source',
    patterns: ['src/**/*.ts'],
    ignore: ['src/**/*.d.ts', 'src/**/*.test.ts', 'src/**/*.spec.ts'],
  },
  {
    family: 'plan',
    patterns: ['plans/**/*.md'],
    ignore: ['plans/completed/**'],
  },
  { family: 'completed-plan', patterns: ['plans/completed/**/*.md'] },
  {
    family: 'demo',
    patterns: ['examples/**/README.md', 'examples/**/*.ts'],
    ignore: [
      ...GENERATED_README_IGNORE,
      'examples/**/*.test.ts',
      SESSION_SNAPSHOT_PATH,
    ],
  },
  {
    family: 'test',
    patterns: ['examples/**/*.test.ts', 'benchmarks/**/*.test.ts'],
  },
  {
    family: 'root-doc',
    patterns: ['README.md', 'STYLEGUIDE.md', 'CONTRIBUTING.md'],
  },
  {
    family: 'skill',
    patterns: ['.github/skills/**/SKILL.md'],
  },
  {
    family: 'agent',
    patterns: ['.github/agents/*.agent.md'],
  },
  {
    family: 'copilot-instructions',
    patterns: ['.github/copilot-instructions.md'],
  },
];

/**
 * The published corpus family names, in publication order.
 *
 * @type {ReadonlyArray<string>}
 */
export const CORPUS_FAMILY_NAMES = CORPUS_SOURCES.map((source) => source.family);
