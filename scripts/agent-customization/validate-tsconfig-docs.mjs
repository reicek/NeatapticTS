#!/usr/bin/env node
import fg from 'fast-glob';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { repoRoot } from './customization-utils.mjs';

const TSCONFIG_DOCS_PATH = path.join(repoRoot, 'tsconfig.docs.json');
const FIX_HINT =
  'Update tsconfig.docs.json include globs or restore the missing paths before running npm run docs.';

function parseArgs(argv) {
  return {
    help: argv.includes('--help') || argv.includes('-h'),
    json: argv.includes('--json'),
  };
}

function printUsage() {
  console.log(
    [
      'Validate docs TSConfig include globs',
      '',
      'Usage:',
      '  node scripts/agent-customization/validate-tsconfig-docs.mjs [--json]',
      '  node scripts/agent-customization/validate-tsconfig-docs.mjs --help',
      '',
      'Options:',
      '  --json  Emit machine-readable JSON.',
    ].join('\n'),
  );
}

export async function validateTsconfigDocs() {
  const tsconfigPayload = JSON.parse(
    await readFile(TSCONFIG_DOCS_PATH, 'utf8'),
  );
  const checkedPaths = Array.isArray(tsconfigPayload.include)
    ? tsconfigPayload.include.filter((value) => typeof value === 'string')
    : [];
  const missingPaths = [];

  for (const includePattern of checkedPaths) {
    const matches = await fg(includePattern, {
      cwd: repoRoot,
      absolute: false,
      dot: true,
    });

    if (matches.length === 0) {
      missingPaths.push(includePattern);
    }
  }

  return {
    pass: missingPaths.length === 0,
    checked_paths: checkedPaths,
    missing_paths: missingPaths,
    fixHint: missingPaths.length === 0 ? null : FIX_HINT,
  };
}

function writeOutput(result, json) {
  if (json) {
    console.log(JSON.stringify(result, null, 2));
    return;
  }

  console.log(
    result.pass ? 'PASS validate-tsconfig-docs' : 'FAIL validate-tsconfig-docs',
  );
  if (!result.pass) console.log(`fixHint: ${result.fixHint}`);
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printUsage();
    return;
  }

  try {
    const result = await validateTsconfigDocs();
    writeOutput(result, options.json);
    process.exitCode = result.pass ? 0 : 1;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const failure = {
      pass: false,
      checked_paths: [],
      missing_paths: [],
      fixHint: FIX_HINT,
      error: message,
    };
    writeOutput(failure, options.json);
    process.exitCode = 1;
  }
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href
) {
  await main();
}
