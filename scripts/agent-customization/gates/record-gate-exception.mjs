#!/usr/bin/env node
/**
 * record-gate-exception — Record a gate exception event and return its structured fields.
 *
 * Accepts gate exception details via CLI arguments and writes a structured exception
 * record to stdout. Optionally appends the record to the learning log as a JSONL event.
 *
 * Output contract:
 *   { "gate-id": string, "exception-evidence": object,
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
import { countTrailingGateFailures, loadLearningLogEvents } from '../enforcement/runtime-enforcement.mjs';

const LEARNING_LOG_PATH = path.join(repoRoot, '.github', 'ai-learning', 'learning-log.jsonl');

// Step 1: Parse arguments.
const args = parseExceptionArgs(process.argv.slice(2));

try {
  // Step 2: Build the exception record.
  const record = buildExceptionRecord(args);

  // Step 3: Output JSON to stdout.
  console.log(JSON.stringify(record, null, 2));

  // Step 4: Append to learning log (best-effort; does not affect exit code).
  await appendLearningEvent(record).catch(() => {
    /* Swallow append errors to keep helper non-blocking. */
  });

  await appendEscalationEventIfNeeded(record).catch(() => {
    /* Keep escalation logging best-effort on the exception path. */
  });

  process.exitCode = 0;
} catch (error) {
  const message = error instanceof Error ? error.message : String(error);
  if (args.json) {
    console.log(JSON.stringify({ ok: false, error: message }, null, 2));
  } else {
    console.error(message);
  }

  process.exitCode = 1;
}

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
 * @returns {{ "gate-id": string, "exception-evidence": object, agent: string, "session-id": string }}
 */
function buildExceptionRecord(args) {
  const gateId = requireNonEmptyField(args.gateId, 'gate-id');
  const agent = requireNonEmptyField(args.agent, 'agent');
  const sessionId = requireNonEmptyField(args.sessionId, 'session-id');

  return {
    'gate-id': gateId,
    'exception-evidence': args.evidence,
    agent,
    'session-id': sessionId,
  };
}

function requireNonEmptyField(value, fieldName) {
  const normalizedValue = typeof value === 'string' ? value.trim() : '';
  if (!normalizedValue) {
    throw new Error(`Missing required non-empty --${fieldName} value.`);
  }

  return normalizedValue;
}

/**
 * Append the exception record to the learning log as a JSONL event.
 * @param {object} record - The gate exception record.
 * @returns {Promise<void>}
 */
async function appendLearningEvent(record) {
  const event = JSON.stringify({
    eventType: 'gate-exception',
    category: 'gate-exception',
    gateId: record['gate-id'],
    agent: record.agent,
    sessionId: record['session-id'],
    exceptionEvidence: record['exception-evidence'],
  });
  await appendFile(LEARNING_LOG_PATH, event + '\n', 'utf8');
}

async function appendEscalationEventIfNeeded(record) {
  const sessionId = String(record['session-id'] ?? '').trim();
  if (!sessionId) {
    return;
  }

  const events = await loadLearningLogEvents();
  const failureCount = countTrailingGateFailures(events, sessionId);
  if (failureCount !== 3) {
    return;
  }

  const escalationEvent = JSON.stringify({
    eventType: 'gate-escalation',
    category: 'gate-escalation',
    gateId: record['gate-id'],
    agent: record.agent,
    sessionId,
    failureCount,
    suggestedAgent: '00-helping',
    reason: 'Three consecutive gate failures reached the escalation threshold.',
  });
  await appendFile(LEARNING_LOG_PATH, escalationEvent + '\n', 'utf8');
}
