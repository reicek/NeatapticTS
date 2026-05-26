#!/usr/bin/env node
/**
 * record-gate-exception — Record a gate exception event and return its structured fields.
 *
 * Accepts gate exception details via CLI arguments and writes a structured exception
 * record to stdout. Optionally appends the record to the learning log as a JSONL event.
 *
 * Output contract:
 *   { timestamp: ISO-8601 string, "gate-id": string, "exception-evidence": object,
 *     agent: string, "session-id": string }
 *
 * Usage:
 *   node scripts/agent-customization/gates/record-gate-exception.mjs \
 *     --json \
 *     --gate-id=<gate-id> \
 *     --agent=<agent-name> \
 *     --session-id=<session-id> \
 *     --evidence=<json-string>
 */

import { appendFile } from 'node:fs/promises';
import path from 'node:path';
import { repoRoot } from '../customization-utils.mjs';

const LEARNING_LOG_PATH = path.join(repoRoot, '.github', 'ai-learning', 'learning-log.jsonl');

// Step 1: Parse arguments.
const args = parseExceptionArgs(process.argv.slice(2));

// Step 2: Build the exception record.
const record = buildExceptionRecord(args);

// Step 3: Output JSON to stdout.
console.log(JSON.stringify(record, null, 2));

// Step 4: Append to learning log (best-effort; does not affect exit code).
await appendLearningEvent(record).catch(() => {
  /* Swallow append errors to keep helper non-blocking. */
});

process.exitCode = 0;

// ---------------------------------------------------------------------------

/**
 * Parse exception CLI arguments into a structured options object.
 * @param {string[]} argv - Raw process.argv slice.
 * @returns {{ json: boolean, gateId: string, agent: string, sessionId: string, evidence: object }}
 */
function parseExceptionArgs(argv) {
  const options = {
    json: false,
    gateId: '',
    agent: '',
    sessionId: '',
    evidence: {},
  };

  for (const rawArg of argv) {
    if (rawArg === '--json') {
      options.json = true;
    } else if (rawArg.startsWith('--gate-id=')) {
      options.gateId = rawArg.slice('--gate-id='.length);
    } else if (rawArg.startsWith('--agent=')) {
      options.agent = rawArg.slice('--agent='.length);
    } else if (rawArg.startsWith('--session-id=')) {
      options.sessionId = rawArg.slice('--session-id='.length);
    } else if (rawArg.startsWith('--evidence=')) {
      const raw = rawArg.slice('--evidence='.length);
      try {
        options.evidence = JSON.parse(raw);
      } catch {
        options.evidence = { rawEvidence: raw };
      }
    }
  }

  return options;
}

/**
 * Build the structured gate-exception record from parsed arguments.
 * @param {{ gateId: string, agent: string, sessionId: string, evidence: object }} args
 * @returns {{ timestamp: string, "gate-id": string, "exception-evidence": object, agent: string, "session-id": string }}
 */
function buildExceptionRecord(args) {
  return {
    timestamp: new Date().toISOString(),
    'gate-id': args.gateId,
    'exception-evidence': args.evidence,
    agent: args.agent,
    'session-id': args.sessionId,
  };
}

/**
 * Append the exception record to the learning log as a JSONL event.
 * @param {object} record - The gate exception record.
 * @returns {Promise<void>}
 */
async function appendLearningEvent(record) {
  const event = JSON.stringify({
    timestamp: record.timestamp,
    eventType: 'gate-exception',
    category: 'gate-exception',
    gateId: record['gate-id'],
    agent: record.agent,
    sessionId: record['session-id'],
    exceptionEvidence: record['exception-evidence'],
  });
  await appendFile(LEARNING_LOG_PATH, event + '\n', 'utf8');
}
