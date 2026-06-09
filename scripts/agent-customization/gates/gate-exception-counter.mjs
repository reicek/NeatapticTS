#!/usr/bin/env node
/**
 * gate-exception-counter — Evaluate the three-exceptions escalation rule.
 *
 * Supports both the legacy explicit `--failure-count=<number>` mode and the
 * durable learning-log mode used by strict runtime enforcement. In durable mode,
 * the helper reconstructs the trailing run of gate failures for the current
 * session from `.github/ai-learning/learning-log.jsonl`, resetting the streak on
 * `runtime-action-prepass` or `runtime-action-postpass` events.
 */

import {
  countTrailingGateFailures,
  loadLearningLogEvents,
} from '../enforcement/runtime-enforcement.mjs';

const args = parseCounterArgs(process.argv.slice(2));
const result = await evaluateEscalation(args);

console.log(JSON.stringify(result, null, 2));

process.exitCode = 0;

// ---------------------------------------------------------------------------

/**
 * Parse counter CLI arguments into a structured options object.
 * @param {string[]} argv - Raw process.argv slice.
 * @returns {{ json: boolean, gateId: string, agent: string, sessionId: string, failureCount: number, deriveFromLearningLog: boolean }}
 */
function parseCounterArgs(argv) {
  const options = {
    json: false,
    gateId: '',
    agent: '',
    sessionId: '',
    failureCount: 0,
    deriveFromLearningLog: false,
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
    } else if (rawArg.startsWith('--failure-count=')) {
      const raw = rawArg.slice('--failure-count='.length);
      const parsed = parseInt(raw, 10);
      options.failureCount = isNaN(parsed) ? 0 : parsed;
    } else if (rawArg === '--derive-from-learning-log') {
      options.deriveFromLearningLog = true;
    }
  }

  return options;
}

/**
 * Evaluate whether the three-exceptions escalation threshold has been reached.
 *
 * @param {{ gateId: string, agent: string, sessionId: string, failureCount: number, deriveFromLearningLog: boolean }} args
 * @returns {Promise<{ escalationTriggered: boolean, suggestedAgent: string | null, sessionId: string, failureCount: number, source: string }>}
 */
async function evaluateEscalation(args) {
  const escalationThreshold = 3;
  const failureCount = args.deriveFromLearningLog
    ? await deriveFailureCountFromLearningLog(args.sessionId)
    : args.failureCount;
  const escalationTriggered = failureCount >= escalationThreshold;

  return {
    escalationTriggered,
    suggestedAgent: escalationTriggered ? '00-helping' : null,
    sessionId: args.sessionId,
    failureCount,
    source: args.deriveFromLearningLog ? 'learning-log' : 'explicit',
  };
}

async function deriveFailureCountFromLearningLog(sessionId) {
  const normalizedSessionId =
    typeof sessionId === 'string' ? sessionId.trim() : '';
  if (!normalizedSessionId) {
    return 0;
  }

  const events = await loadLearningLogEvents();
  return countTrailingGateFailures(events, normalizedSessionId);
}
