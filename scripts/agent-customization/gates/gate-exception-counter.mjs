#!/usr/bin/env node
/**
 * gate-exception-counter — Evaluate the three-exceptions escalation rule.
 *
 * Accepts a `--failure-count` argument representing the number of consecutive gate
 * failures in a session. Returns whether the three-exceptions escalation threshold
 * has been reached and, if so, names `00-helping` as the suggested escalation target.
 *
 * Escalation rule:
 *   - failure-count < 3 → escalationTriggered: false
 *   - failure-count >= 3 → escalationTriggered: true, suggestedAgent: "00-helping"
 *   - failure-count === 0 → escalationTriggered: false (counter reset)
 *
 * Output contract:
 *   { escalationTriggered: boolean, suggestedAgent: string | null,
 *     sessionId: string, failureCount: number }
 *
 * Usage:
 *   node scripts/agent-customization/gates/gate-exception-counter.mjs \
 *     --json \
 *     --gate-id=<gate-id> \
 *     --agent=<agent-name> \
 *     --session-id=<session-id> \
 *     --failure-count=<number>
 */

// Step 1: Parse arguments.
const args = parseCounterArgs(process.argv.slice(2));

// Step 2: Evaluate the escalation threshold.
const result = evaluateEscalation(args);

// Step 3: Output JSON to stdout.
console.log(JSON.stringify(result, null, 2));

process.exitCode = 0;

// ---------------------------------------------------------------------------

/**
 * Parse counter CLI arguments into a structured options object.
 * @param {string[]} argv - Raw process.argv slice.
 * @returns {{ json: boolean, gateId: string, agent: string, sessionId: string, failureCount: number }}
 */
function parseCounterArgs(argv) {
  const options = {
    json: false,
    gateId: '',
    agent: '',
    sessionId: '',
    failureCount: 0,
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
    }
  }

  return options;
}

/**
 * Evaluate whether the three-exceptions escalation threshold has been reached.
 *
 * @param {{ gateId: string, agent: string, sessionId: string, failureCount: number }} args
 * @returns {{ escalationTriggered: boolean, suggestedAgent: string | null, sessionId: string, failureCount: number }}
 */
function evaluateEscalation(args) {
  const escalationThreshold = 3;
  const escalationTriggered = args.failureCount >= escalationThreshold;

  return {
    escalationTriggered,
    suggestedAgent: escalationTriggered ? '00-helping' : null,
    sessionId: args.sessionId,
    failureCount: args.failureCount,
  };
}
