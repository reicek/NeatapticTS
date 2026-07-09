#!/usr/bin/env node
/**
 * Tier-1 gate: learning-event
 *
 * Checks that .github/ai-learning/learning-log.jsonl exists and contains at
 * least one parseable event entry, satisfying the requirement that gate
 * exceptions and cross-tier helper calls are recorded before phase completion.
 *
 * Gate contract: { pass: boolean, evidence: object, fixHint: string, owner: string }
 *
 * Usage:
 *   node scripts/agent-customization/gates/learning-event.gate.mjs [--json]
 */

import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { parseArgs, repoRoot } from '../customization-utils.mjs';

const LEARNING_LOG_PATH = process.env.NEATAPTIC_LEARNING_LOG_PATH
  ? path.resolve(process.env.NEATAPTIC_LEARNING_LOG_PATH)
  : path.join(repoRoot, '.github', 'ai-learning', 'learning-log.jsonl');

const options = parseArgs(process.argv.slice(2));

const result = await runLearningEventGate();

if (options.json) {
  console.log(JSON.stringify(result, null, 2));
} else {
  console.log(result.pass ? 'PASS' : 'FAIL', 'learning-event gate');
  if (!result.pass) console.log('fixHint:', result.fixHint);
}

process.exitCode = result.pass ? 0 : 1;

// ---------------------------------------------------------------------------

async function runLearningEventGate() {
  // Step 1: Attempt to read the learning event log.
  let text = '';
  try {
    text = await readFile(LEARNING_LOG_PATH, 'utf8');
  } catch {
    return {
      pass: false,
      evidence: { exists: false, path: LEARNING_LOG_PATH },
      fixHint: `Create ${LEARNING_LOG_PATH} and append at least one JSONL learning event before gating.`,
      owner: '.github/ai-learning/learning-log.jsonl',
    };
  }

  // Step 2: Parse each non-empty line as a JSONL event.
  const rawLines = text.split('\n').filter((line) => line.trim().length > 0);
  const events = [];
  for (const line of rawLines) {
    try {
      events.push(JSON.parse(line));
    } catch {
      // Skip malformed lines; presence of valid events is the requirement.
    }
  }

  const pass = events.length > 0;

  // Step 3: Collect category summary for evidence.
  const categories = [
    ...new Set(
      events.map((event) => event.eventType ?? event.category).filter(Boolean),
    ),
  ];

  return {
    pass,
    evidence: {
      exists: true,
      path: LEARNING_LOG_PATH,
      eventCount: events.length,
      rawLineCount: rawLines.length,
      categories,
    },
    fixHint: pass
      ? 'Learning event log exists and contains at least one valid event.'
      : `Append at least one JSONL event to ${LEARNING_LOG_PATH} before gating.`,
    owner: '.github/ai-learning/learning-log.jsonl',
  };
}
