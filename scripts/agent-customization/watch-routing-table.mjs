#!/usr/bin/env node
/**
 * @description Watch `.github/agents` and `.github/skills` and regenerate the
 * canonical agent-skill routing table whenever a source file is saved.
 *
 * @param {boolean} [--help] - Show help and exit.
 * @param {number}  [--debounce-ms=<n>] - Delay before regeneration after a file event.
 *
 * @returns {void} Starts a long-running file watcher until interrupted.
 */
import { watch } from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { pathToFileURL } from 'node:url';

import { parseArgs, repoRoot } from './customization-utils.mjs';

const DEFAULT_DEBOUNCE_MS = 250;
const WATCH_DIRECTORIES = ['.github/agents', '.github/skills'];
const GENERATOR_PATH = path.join(repoRoot, 'scripts', 'agent-customization', 'generate-agent-skill-routing-table.mjs');

function parseCliOptions(argv) {
  const base = parseArgs(argv);
  const debounceArgument = argv.find((argument) => argument.startsWith('--debounce-ms='));

  return {
    ...base,
    debounceMs: debounceArgument ? Math.max(0, Number(debounceArgument.slice('--debounce-ms='.length)) || DEFAULT_DEBOUNCE_MS) : DEFAULT_DEBOUNCE_MS,
  };
}

function printUsage() {
  console.log([
    'Routing-table watch utility',
    '',
    'Usage:',
    '  node scripts/agent-customization/watch-routing-table.mjs [--debounce-ms=<n>]',
    '  node scripts/agent-customization/watch-routing-table.mjs --help',
    '',
    'Options:',
    '  --debounce-ms=<n>  Delay before regeneration after a watched file event.',
    '  --help             Show this help.',
  ].join('\n'));
}

function shouldRegenerate(relativePath = '') {
  return relativePath.endsWith('.agent.md') || relativePath.endsWith('SKILL.md');
}

function runGenerator(triggerLabel) {
  const result = spawnSync(process.execPath, [GENERATOR_PATH, '--json'], {
    cwd: repoRoot,
    encoding: 'utf8',
  });

  if (result.status !== 0) {
    const stderr = result.stderr?.trim();
    const stdout = result.stdout?.trim();
    console.error(`[routing-table-watch] regeneration failed after ${triggerLabel}`);
    if (stderr) console.error(stderr);
    if (!stderr && stdout) console.error(stdout);
    return;
  }

  const report = tryParseJson(result.stdout);
  if (!report) {
    console.error('[routing-table-watch] regeneration returned unreadable JSON output.');
    return;
  }

  const changedSuffix = report.changed ? 'updated' : 'already fresh';
  console.error(`[routing-table-watch] ${changedSuffix} after ${triggerLabel}`);
}

function tryParseJson(value) {
  if (typeof value !== 'string' || !value.trim()) {
    return null;
  }

  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

function startWatcher(relativeDirectory, debounceMs, scheduleRefresh) {
  const absoluteDirectory = path.join(repoRoot, relativeDirectory);

  return watch(absoluteDirectory, { recursive: true }, (_eventType, filename) => {
    const relativePath = typeof filename === 'string'
      ? path.posix.join(relativeDirectory.replace(/\\/g, '/'), filename.replace(/\\/g, '/'))
      : relativeDirectory;

    if (!shouldRegenerate(relativePath)) {
      return;
    }

    scheduleRefresh(relativePath, debounceMs);
  });
}

async function main() {
  const options = parseCliOptions(process.argv.slice(2));

  if (options.help) {
    printUsage();
    return;
  }

  let debounceHandle;
  const watchers = [];
  const scheduleRefresh = (relativePath, debounceMs) => {
    clearTimeout(debounceHandle);
    debounceHandle = setTimeout(() => {
      runGenerator(relativePath);
    }, debounceMs);
  };

  runGenerator('startup');
  for (const directory of WATCH_DIRECTORIES) {
    watchers.push(startWatcher(directory, options.debounceMs, scheduleRefresh));
  }

  console.error('[routing-table-watch] watching .github/agents and .github/skills for changes');

  const shutdown = () => {
    clearTimeout(debounceHandle);
    for (const watcher of watchers) {
      watcher.close();
    }
    process.exit(0);
  };

  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  await main();
}
