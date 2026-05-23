import path from 'node:path';
import { repoRoot } from './init-schema.mjs';

export function parseCliArgs(argv) {
  const flags = { _: [] };

  for (let argumentIndex = 0; argumentIndex < argv.length; argumentIndex += 1) {
    const argument = argv[argumentIndex];
    if (!argument.startsWith('--')) {
      flags._.push(argument);
      continue;
    }

    const [rawKey, inlineValue] = argument.slice(2).split('=', 2);
    const nextValue = argv[argumentIndex + 1];
    const hasSeparateValue = nextValue !== undefined && !nextValue.startsWith('--');
    flags[rawKey] = inlineValue ?? (hasSeparateValue ? nextValue : true);
    if (inlineValue === undefined && hasSeparateValue) argumentIndex += 1;
  }

  return flags;
}

export function printHelp({ title, usage, options }) {
  console.log([title, '', `Usage: ${usage}`, '', 'Options:', ...options.map((option) => `  ${option}`)].join('\n'));
}

export function writeJsonOrText(payload, json, formatText) {
  if (json) {
    console.log(JSON.stringify(payload, null, 2));
    return;
  }

  console.log(formatText(payload));
}

export function toRepoRelative(filePath) {
  return path.relative(repoRoot, filePath).replaceAll(path.sep, '/');
}

export function fail(message, json = false, details = {}) {
  if (json) console.log(JSON.stringify({ pass: false, ok: false, error: message, ...details }, null, 2));
  else console.error(message);
  process.exitCode = 1;
}