#!/usr/bin/env node
/**
 * @module folder-quality-gate
 * @description Standard gate wrapper around `scripts/folder-quality-metrics.mjs`.
 *
 * Usage:
 *   node scripts/agent-customization/gates/folder-quality.gate.mjs --folder=<path> [--json]
 *   node scripts/agent-customization/gates/folder-quality.gate.mjs --help
 *
 * Exit codes:
 *   0: gate passed
 *   1: gate failed or arguments were invalid
 */

import { pathToFileURL } from 'node:url';

import { runFolderQualityMetrics } from '../../folder-quality-metrics.mjs';

const OWNER = '05-green-testing';

function parseArgs(argv) {
  const options = {
    folder: null,
    help: false,
    json: false,
  };

  for (const rawArgument of argv) {
    if (rawArgument === '--help' || rawArgument === '-h') {
      options.help = true;
      continue;
    }

    if (rawArgument === '--json') {
      options.json = true;
      continue;
    }

    if (rawArgument.startsWith('--folder=')) {
      options.folder = rawArgument.slice('--folder='.length);
      continue;
    }

    throw new Error(`Unknown argument: ${rawArgument}`);
  }

  return options;
}

function printUsage() {
  console.log(
    [
      'Folder-quality gate',
      '',
      'Usage:',
      '  node scripts/agent-customization/gates/folder-quality.gate.mjs --folder=<path> [--json]',
      '  node scripts/agent-customization/gates/folder-quality.gate.mjs --help',
      '',
      'Flags:',
      '  --folder=<path>  Repo-relative folder to check.',
      '  --json           Emit the standard gate contract to stdout as JSON.',
      '  --help           Show usage and exit codes.',
      '',
      'Exit codes:',
      '  0  Gate passed',
      '  1  Gate failed or arguments were invalid',
    ].join('\n'),
  );
}

/**
 * Execute the folder-quality gate contract.
 *
 * @param {{ folderPath: string }} options - Gate options.
 * @returns {Promise<{ pass: boolean, evidence: object, fixHint: string | null, owner: string, schema_version: number }>} Standard gate contract.
 */
export async function runFolderQualityGate({ folderPath }) {
  const metricsReport = await runFolderQualityMetrics({ folderPath });

  return {
    schema_version: 1,
    pass: metricsReport.pass,
    evidence: {
      evidence: metricsReport.evidence,
      folderChecked: metricsReport.folderChecked,
      smellCount: metricsReport.smells.length,
      smells: metricsReport.smells,
    },
    fixHint: metricsReport.pass
      ? null
      : `Run npm run quality:folder -- --folder=${metricsReport.folderChecked} and resolve the reported smells before marking the folder green.`,
    owner: OWNER,
  };
}

async function main() {
  try {
    const options = parseArgs(process.argv.slice(2));
    if (options.help) {
      printUsage();
      return;
    }

    if (!options.folder) {
      throw new Error('--folder=<path> is required.');
    }

    const report = await runFolderQualityGate({ folderPath: options.folder });
    if (options.json) {
      console.log(JSON.stringify(report, null, 2));
    } else {
      console.log(
        report.pass ? 'PASS folder-quality gate' : 'FAIL folder-quality gate',
      );
      if (!report.pass) {
        console.log(`fixHint: ${report.fixHint}`);
      }
    }

    process.exitCode = report.pass ? 0 : 1;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    process.stderr.write(`${message}\n`);
    process.exitCode = 1;
  }
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href
) {
  await main();
}
