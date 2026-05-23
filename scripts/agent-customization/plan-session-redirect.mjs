#!/usr/bin/env node
import { existsSync } from 'node:fs';
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { repoRoot } from './customization-utils.mjs';

const OVERRIDE_PATH = path.join(repoRoot, 'data', 'mcp-session-override.json');
const PLANS_ROOT = path.join(repoRoot, 'plans');

function parseArgs(argv) {
  return {
    help: argv.includes('--help') || argv.includes('-h'),
    json: argv.includes('--json'),
    clear: argv.includes('--clear'),
    plan: argv.find((argument) => argument.startsWith('--plan='))?.slice('--plan='.length),
  };
}

function printUsage() {
  console.log([
    'Plan session redirect',
    '',
    'Usage:',
    '  node scripts/agent-customization/plan-session-redirect.mjs --plan=<path> [--json]',
    '  node scripts/agent-customization/plan-session-redirect.mjs --clear [--json]',
    '  node scripts/agent-customization/plan-session-redirect.mjs --help',
    '',
    'Options:',
    '  --plan=<path>  Plan path to persist in data/mcp-session-override.json.',
    '  --clear        Remove the override file. Idempotent when it is already absent.',
    '  --json         Emit machine-readable JSON.',
  ].join('\n'));
}

async function writeSessionOverride(planPath) {
  const resolvedPlanPath = resolvePlanPath(planPath);
  const payload = {
    plan_path: resolvedPlanPath,
  };

  await mkdir(path.dirname(OVERRIDE_PATH), { recursive: true });
  await writeFile(OVERRIDE_PATH, `${JSON.stringify(payload, null, 2)}\n`, 'utf8');

  const confirmedPayload = JSON.parse(await readFile(OVERRIDE_PATH, 'utf8'));
  if (confirmedPayload.plan_path !== resolvedPlanPath) {
    throw new Error('Session override readback did not match the requested plan_path.');
  }

  return {
    pass: true,
    override_written: true,
    plan_path: resolvedPlanPath,
  };
}

async function clearSessionOverride() {
  const wasPresent = existsSync(OVERRIDE_PATH);
  if (wasPresent) {
    await rm(OVERRIDE_PATH, { force: true });
  }

  return {
    pass: true,
    override_cleared: true,
    was_present: wasPresent,
  };
}

function resolvePlanPath(candidatePath) {
  if (typeof candidatePath !== 'string' || !candidatePath.trim()) {
    throw new Error('Provide --plan=<path> within plans/.');
  }

  const absolutePlanPath = path.isAbsolute(candidatePath)
    ? path.normalize(candidatePath)
    : path.resolve(repoRoot, candidatePath);
  const relativeToPlans = path.relative(PLANS_ROOT, absolutePlanPath);
  const staysWithinPlans = relativeToPlans !== ''
    && !relativeToPlans.startsWith('..')
    && !path.isAbsolute(relativeToPlans);

  if (!staysWithinPlans) {
    throw new Error(`Plan path must resolve within plans/. Received: ${candidatePath}`);
  }

  if (!existsSync(absolutePlanPath)) {
    throw new Error(`Plan file not found: ${normalizePath(path.relative(repoRoot, absolutePlanPath))}`);
  }

  return normalizePath(path.relative(repoRoot, absolutePlanPath));
}

function normalizePath(value) {
  return value.replaceAll(path.sep, '/');
}

function writeOutput(result, json) {
  if (json) {
    console.log(JSON.stringify(result, null, 2));
    return;
  }

  console.log(result.pass ? 'PASS plan-session-redirect' : 'FAIL plan-session-redirect');
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printUsage();
    return;
  }

  try {
    if (options.clear) {
      writeOutput(await clearSessionOverride(), options.json);
      return;
    }

    writeOutput(await writeSessionOverride(options.plan), options.json);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const failure = { pass: false, fixHint: 'Use --plan=<path> within plans/ or --clear to remove the override.' };
    if (options.json) {
      console.log(JSON.stringify({ ...failure, error: message }, null, 2));
    } else {
      console.error(message);
    }
    process.exitCode = 1;
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}