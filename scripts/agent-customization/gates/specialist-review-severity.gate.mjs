#!/usr/bin/env node
/**
 * Tier-1 gate: specialist-review-severity
 *
 * Classifies a proposed fix as TRIVIAL or FULL based on the changed files.
 * TRIVIAL fixes (test-only, JSDoc-only, formatting-only, skill/policy .md)
 * skip specialist review entirely. FULL fixes that touch runtime logic
 * under src/ or examples/ require exactly 1 specialist reviewer.
 *
 * Gate contract: { pass: boolean, evidence: object, fixHint: string, owner: string }
 *
 * Library usage:
 *   import { classifySeverity } from './specialist-review-severity.gate.mjs';
 *   const result = classifySeverity(['testing/foo.test.ts']);
 *
 * CLI usage:
 *   node scripts/agent-customization/gates/specialist-review-severity.gate.mjs --json --input=src/foo.ts,testing/foo.test.ts
 */

import path from 'node:path';
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

import { parseArgs } from '../customization-utils.mjs';

/**
 * Matches test and spec files by extension.
 * Recognizes `.test.{ts,js,mjs,cjs}` and `.spec.{ts,js,mjs,cjs}` paths.
 */
const TEST_PATTERN = /\.(test|spec)\.(ts|js|mjs|cjs)$/i;

/**
 * Matches Markdown documentation files.
 * Used to classify JSDoc/README-only edits as trivial.
 */
const MARKDOWN_PATTERN = /\.md$/i;

/**
 * Matches formatting and editor configuration files.
 * These files only affect style conventions, not runtime behavior.
 */
const FORMATTING_CONFIG_PATTERN =
  /(^|\/)(\.prettierrc.*|\.prettierignore|\.editorconfig|\.gitattributes|\.eslintignore|\.gitignore|\.npmrc|\.nvmrc)$/;

/**
 * Matches package-manager lock files.
 * Lock file changes are deterministic metadata, not runtime logic.
 */
const LOCKFILE_PATTERN =
  /(^|\/)(package-lock\.json|yarn\.lock|pnpm-lock\.yaml)$/;

/**
 * Determine whether a single changed file is trivial by path inspection.
 *
 * A trivial file is one whose changes are unlikely to alter runtime behavior:
 * test files, Markdown/JSDoc-only surfaces, formatting configuration, and lock
 * files. Source logic under src/, examples/, or benchmarks/ is never trivial.
 *
 * @param filePath - repository-relative file path.
 * @returns true when the file is trivial by path heuristic.
 */
function isTrivialFile(filePath) {
  if (TEST_PATTERN.test(filePath)) return true;
  if (MARKDOWN_PATTERN.test(filePath)) return true;
  if (FORMATTING_CONFIG_PATTERN.test(filePath)) return true;
  if (LOCKFILE_PATTERN.test(filePath)) return true;

  return false;
}

/**
 * Classify the severity of a set of changed files.
 *
 * Returns TRIVIAL when every changed file is a test file, Markdown/JSDoc surface,
 * formatting config, or lock file. Returns FULL as soon as one file touches
 * runtime logic under src/, examples/, or benchmarks/. Mixed changes include
 * a partitioned list of trivial and non-trivial files.
 *
 * @param changedFiles - repository-relative file paths changed in the fix.
 * @returns Classification result with severity and optional file partitions.
 */
export function classifySeverity(changedFiles) {
  if (!Array.isArray(changedFiles)) {
    throw new TypeError('classifySeverity expects an array of file paths');
  }

  if (changedFiles.length === 0) {
    return {
      severity: 'TRIVIAL',
      specialistCount: 0,
      trivialFiles: [],
      nonTrivialFiles: [],
    };
  }

  const trivialFiles = [];
  const nonTrivialFiles = [];

  for (const filePath of changedFiles) {
    if (typeof filePath !== 'string') {
      throw new TypeError('classifySeverity expects file paths to be strings');
    }

    if (isTrivialFile(filePath)) {
      trivialFiles.push(filePath);
    } else {
      nonTrivialFiles.push(filePath);
    }
  }

  if (nonTrivialFiles.length === 0) {
    return {
      severity: 'TRIVIAL',
      specialistCount: 0,
      trivialFiles,
      nonTrivialFiles,
    };
  }

  return {
    severity: 'FULL',
    specialistCount: 1,
    trivialFiles,
    nonTrivialFiles,
  };
}

const options = parseArgs(process.argv.slice(2));

function printUsage() {
  console.log(
    'specialist-review-severity gate\n\n' +
      'Usage:\n' +
      '  node specialist-review-severity.gate.mjs --input=path1,path2\n' +
      '  node specialist-review-severity.gate.mjs --json --input=@file-list.txt\n\n' +
      'Options:\n' +
      '  --input=<paths>  Comma-separated changed file paths, or @file to read a list.\n' +
      '  --json           Write machine-readable JSON to stdout.\n' +
      '  --help, -h       Show this help text.',
  );
}

function loadChangedFiles(input) {
  if (!input) {
    return [];
  }

  if (input.startsWith('@')) {
    const listPath = input.slice(1);
    const raw = readFileSync(listPath, 'utf8');
    return raw
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line.length > 0 && !line.startsWith('#'));
  }

  return input
    .split(',')
    .map((entry) => entry.trim())
    .filter(Boolean);
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href
) {
  if (options.help) {
    printUsage();
    process.exitCode = 0;
  } else {
    const changedFiles = loadChangedFiles(options.input);
    const classification = classifySeverity(changedFiles);
    const pass = true;

    const reason =
      classification.severity === 'TRIVIAL'
        ? changedFiles.length === 0
          ? 'no changed files supplied'
          : 'all changes are test, doc, formatting, or lock files'
        : 'runtime logic changed under src/, examples/, or benchmarks/';

    const result = {
      pass,
      evidence: {
        gate: 'specialist-review-severity',
        tier: 1,
        classification,
        reason,
      },
      fixHint:
        'TRIVIAL fixes skip specialist review entirely; FULL fixes dispatch 1 specialist before green testing.',
      owner:
        'scripts/agent-customization/gates/specialist-review-severity.gate.mjs',
    };

    if (options.json) {
      console.log(JSON.stringify(result, null, 2));
    } else {
      console.log(
        classification.severity,
        'specialist-review-severity gate',
        `— ${reason}`,
      );
      if (classification.nonTrivialFiles.length > 0) {
        console.log('non-trivial files:');
        for (const file of classification.nonTrivialFiles) {
          console.log(`  ${file}`);
        }
      }
    }

    process.exitCode = 0;
  }
}
