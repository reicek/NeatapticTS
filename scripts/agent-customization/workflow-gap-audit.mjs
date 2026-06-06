#!/usr/bin/env node
/**
 * workflow-gap-audit — Learning-event analytics aggregator for the `00.workflow-gap-audit` flow.
 *
 * Reads gate exception log and optional session store to aggregate:
 *   - Gate failure frequency (real exceptions only; test-session artifacts filtered)
 *   - Escalation count (gate-escalation category events)
 *   - Runtime enforcement evidence (pre/post pass pairs, proof mismatches, bypass signals)
 *   - Agent session counts (from SQLite session store when `--db` is provided)
 *   - Agent drift sessions (sessions without a named flow ID in summary)
 *   - Underused flows (flow IDs absent from recent session summaries)
 *
 * Output contract (matches 00.workflow-gap-audit analytics contract):
 * ```jsonc
 * {
 *   "reportDate": "<ISO-8601>",
 *   "windowDays": 7,
 *   "gateFailureFrequency": [{ "gateId": "...", "failureCount": 0, "affectedAgents": [] }],
 *   "escalationCount": 0,
 *   "runtimeEnforcementEvidence": {
 *     "preActionPasses": 0,
 *     "postActionPasses": 0,
 *     "proofMismatches": 0,
 *     "blockedActions": 0,
 *     "missingPostActionPairs": [],
 *     "postWithoutPrePairs": []
 *   },
 *   "agentSessionCounts": [{ "agentName": "...", "sessionCount": 0 }],
 *   "agentDriftSessions": [{ "sessionId": "...", "agentName": "...", "summary": "..." }],
 *   "underusedFlows": [{ "flowId": "...", "mentionCount": 0 }],
 *   "topFailingGate": "<id> | null",
 *   "recommendedActions": ["..."]
 * }
 * ```
 *
 * Test-session filter: events with `sessionId === "test-session-001"` or
 * `exceptionEvidence.reason === "test-red-phase"` are excluded from all aggregations.
 *
 * Usage:
 *   node scripts/agent-customization/workflow-gap-audit.mjs [--json] [--window=7] [--db=<sqlite-path>] [--help]
 */

import { readdir, readFile } from 'node:fs/promises';
import path from 'node:path';
import { parseArgs, repoRoot } from './customization-utils.mjs';

const LEARNING_LOG_PATH = '.github/ai-learning/learning-log.jsonl';
const FLOWS_DIR = '.github/flows';

/** Named-flow ID pattern: two digits, dot, kebab-case name (e.g. `04.scoped-fix`). */
const FLOW_ID_PATTERN = /\b\d{2}\.[a-z][a-z0-9-]*\b/g;

/** Session IDs that are test artifacts and must be excluded from real aggregations. */
const TEST_SESSION_IDS = new Set(['test-session-001']);

/** Evidence reasons that mark a gate exception as a test artifact. */
const TEST_EVIDENCE_REASONS = new Set(['test-red-phase']);

const options = parseCliOptions(process.argv.slice(2));

if (options.help) {
  printUsage();
  process.exit(0);
}

const report = await runWorkflowGapAudit(options);

if (options.json) {
  console.log(JSON.stringify(report, null, 2));
} else {
  printHumanReport(report);
}

process.exitCode = 0;

// ---------------------------------------------------------------------------

/**
 * Parse workflow-gap-audit CLI arguments.
 * @param {string[]} argv - Raw process.argv slice.
 * @returns {{ json: boolean, help: boolean, windowDays: number, dbPath: string | null }}
 */
function parseCliOptions(argv) {
  const base = parseArgs(argv);
  const windowArg = argv.find((argument) => argument.startsWith('--window='));
  const dbArg = argv.find((argument) => argument.startsWith('--db='));

  return {
    ...base,
    windowDays: windowArg ? Math.max(1, parseInt(windowArg.slice('--window='.length), 10) || 7) : 7,
    dbPath: dbArg ? dbArg.slice('--db='.length) : null,
  };
}

/**
 * Main audit runner — aggregates all analytics sources and returns a structured report.
 * @param {{ windowDays: number, dbPath: string | null }} opts - Audit options.
 * @returns {Promise<object>} Structured analytics report.
 */
async function runWorkflowGapAudit(opts) {
  const reportDate = new Date().toISOString();
  const cutoffDate = new Date(Date.now() - opts.windowDays * 24 * 60 * 60 * 1000).toISOString();

  // Step 1: Load and parse the gate exception log.
  const allEvents = await loadLearningLog();

  // Step 2: Filter out test-session artifacts before any aggregation.
  const realEvents = filterTestArtifacts(allEvents);

  // Step 3: Aggregate gate failure frequency from real exceptions only.
  const gateFailureFrequency = aggregateGateFailures(realEvents, cutoffDate);

  // Step 4: Count escalation events within the window.
  const escalationCount = countEscalations(realEvents, cutoffDate);

  // Step 4b: Aggregate runtime enforcement evidence within the window.
  const runtimeEnforcementEvidence = aggregateRuntimeEnforcementEvidence(realEvents, cutoffDate);

  // Step 5: Load session store data (graceful fallback when unavailable or empty).
  const sessionData = await loadSessionStoreData(opts.dbPath, cutoffDate, opts.windowDays);

  // Step 6: Load all registered flow IDs from .github/flows/*.flow.yml.
  const allFlowIds = await loadFlowIds();

  // Step 7: Compute underused flows against session summaries.
  const underusedFlows = computeUnderusedFlows(allFlowIds, sessionData.recentSummaries);

  // Step 8: Build recommended actions from aggregation results.
  const recommendedActions = buildRecommendedActions({
    gateFailureFrequency,
    escalationCount,
    runtimeEnforcementEvidence,
    agentDriftSessions: sessionData.driftSessions,
    underusedFlows,
  });

  return {
    reportDate,
    windowDays: opts.windowDays,
    gateFailureFrequency,
    escalationCount,
    runtimeEnforcementEvidence,
    agentSessionCounts: sessionData.agentSessionCounts,
    agentDriftSessions: sessionData.driftSessions,
    underusedFlows,
    topFailingGate: gateFailureFrequency.at(0)?.gateId ?? null,
    recommendedActions,
  };
}

// ---------------------------------------------------------------------------
// Learning log helpers
// ---------------------------------------------------------------------------

/**
 * Load and parse all JSONL events from the learning log.
 * Returns an empty array if the log file cannot be read.
 * @returns {Promise<object[]>}
 */
async function loadLearningLog() {
  let text = '';
  try {
    text = await readFile(path.join(repoRoot, LEARNING_LOG_PATH), 'utf8');
  } catch {
    return [];
  }

  const events = [];
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      events.push(JSON.parse(trimmed));
    } catch {
      // Skip malformed lines silently.
    }
  }

  return events;
}

/**
 * Remove test-session artifacts from the event list.
 * Filters entries where sessionId is a known test session OR
 * exceptionEvidence.reason is a known test marker.
 * @param {object[]} events - All parsed learning log events.
 * @returns {object[]} Real events only.
 */
function filterTestArtifacts(events) {
  return events.filter((event) => {
    if (TEST_SESSION_IDS.has(event.sessionId)) return false;
    if (TEST_EVIDENCE_REASONS.has(event.exceptionEvidence?.reason)) return false;
    return true;
  });
}

/**
 * Aggregate gate failure frequency from real gate-exception events within the window.
 * Returns list sorted by failure count descending.
 * @param {object[]} realEvents - Filtered real events.
 * @param {string} cutoffDate - ISO-8601 cutoff date string.
 * @returns {{ gateId: string, failureCount: number, affectedAgents: string[] }[]}
 */
function aggregateGateFailures(realEvents, cutoffDate) {
  const failureMap = new Map();

  for (const event of realEvents) {
    if (event.eventType !== 'gate-exception') continue;
    if (typeof event.timestamp === 'string' && event.timestamp < cutoffDate) continue;

    const gateId = event.gateId ?? 'unknown';
    const agent = event.agent ?? 'unknown';

    if (!failureMap.has(gateId)) {
      failureMap.set(gateId, { gateId, failureCount: 0, affectedAgents: new Set() });
    }
    const entry = failureMap.get(gateId);
    entry.failureCount += 1;
    entry.affectedAgents.add(agent);
  }

  // Step: Serialize Set to sorted array for JSON output.
  return [...failureMap.values()]
    .map((entry) => ({
      gateId: entry.gateId,
      failureCount: entry.failureCount,
      affectedAgents: [...entry.affectedAgents].toSorted(),
    }))
    .toSorted((a, b) => b.failureCount - a.failureCount);
}

/**
 * Count gate-escalation events within the window.
 * @param {object[]} realEvents - Filtered real events.
 * @param {string} cutoffDate - ISO-8601 cutoff date string.
 * @returns {number}
 */
function countEscalations(realEvents, cutoffDate) {
  return realEvents.filter(
    (event) =>
      (event.category === 'gate-escalation' || event.eventType === 'gate-escalation') &&
      (typeof event.timestamp !== 'string' || event.timestamp >= cutoffDate),
  ).length;
}

/**
 * Aggregate runtime enforcement events from the learning log.
 *
 * @param {object[]} realEvents
 * @param {string} cutoffDate
 * @returns {{ preActionPasses: number, postActionPasses: number, proofMismatches: number, blockedActions: number, missingPostActionPairs: string[], postWithoutPrePairs: string[] }}
 */
function aggregateRuntimeEnforcementEvidence(realEvents, cutoffDate) {
  const preActionPasses = new Set();
  const postActionPasses = new Set();
  let proofMismatches = 0;
  let blockedActions = 0;

  for (const event of realEvents) {
    if (typeof event.timestamp === 'string' && event.timestamp < cutoffDate) continue;

    if (event.eventType === 'runtime-action-prepass' && typeof event.actionId === 'string') {
      preActionPasses.add(event.actionId);
      continue;
    }

    if (event.eventType === 'runtime-action-postpass' && typeof event.actionId === 'string') {
      postActionPasses.add(event.actionId);
      continue;
    }

    if (event.eventType === 'runtime-proof-mismatch') {
      proofMismatches += 1;
      blockedActions += 1;
    }

    if (event.eventType === 'runtime-action-blocked') {
      blockedActions += 1;
    }
  }

  const missingPostActionPairs = [...preActionPasses]
    .filter((actionId) => !postActionPasses.has(actionId))
    .toSorted();
  const postWithoutPrePairs = [...postActionPasses]
    .filter((actionId) => !preActionPasses.has(actionId))
    .toSorted();

  return {
    preActionPasses: preActionPasses.size,
    postActionPasses: postActionPasses.size,
    proofMismatches,
    blockedActions,
    missingPostActionPairs,
    postWithoutPrePairs,
  };
}

// ---------------------------------------------------------------------------
// Session store helpers
// ---------------------------------------------------------------------------

/**
 * Load session store data: agent counts, drift sessions, and recent summaries.
 * Uses better-sqlite3 when `--db=<path>` is provided; otherwise returns empty arrays.
 * @param {string | null} dbPath - Optional absolute path to the SQLite session store file.
 * @param {string} cutoffDate - ISO-8601 cutoff date string.
 * @param {number} windowDays - Rolling window for underused-flow detection (longer: 30d).
 * @returns {Promise<{ agentSessionCounts: object[], driftSessions: object[], recentSummaries: string[] }>}
 */
async function loadSessionStoreData(dbPath, cutoffDate, windowDays) {
  const emptyResult = { agentSessionCounts: [], driftSessions: [], recentSummaries: [] };

  if (!dbPath) {
    return emptyResult;
  }

  // Attempt to dynamically import better-sqlite3 (optional peer dependency).
  let Database;
  try {
    const module = await import('better-sqlite3');
    Database = module.default ?? module.Database;
  } catch {
    // better-sqlite3 not available; session store aggregations return empty.
    return emptyResult;
  }

  let db;
  try {
    db = new Database(dbPath, { readonly: true, fileMustExist: true });
  } catch {
    return emptyResult;
  }

  try {
    // Agent session counts within the window.
    const agentSessionCounts = db
      .prepare(
        `SELECT agent_name AS agentName, COUNT(*) AS sessionCount
         FROM sessions
         WHERE created_at > :cutoff AND agent_name IS NOT NULL
         GROUP BY agent_name
         ORDER BY sessionCount DESC`,
      )
      .all({ cutoff: cutoffDate });

    // Drift sessions: recent sessions whose summary contains no named flow ID.
    const recentRows = db
      .prepare(
        `SELECT id AS sessionId, agent_name AS agentName, summary
         FROM sessions
         WHERE created_at > :cutoff AND summary IS NOT NULL`,
      )
      .all({ cutoff: cutoffDate });

    const driftSessions = recentRows
      .filter((row) => !FLOW_ID_PATTERN.test(row.summary ?? ''))
      .map((row) => ({
        sessionId: row.sessionId,
        agentName: row.agentName ?? 'unknown',
        summary: (row.summary ?? '').slice(0, 200),
      }));

    // Longer 30-day window for underused-flow detection.
    const longCutoff = new Date(
      Date.now() - Math.max(windowDays, 30) * 24 * 60 * 60 * 1000,
    ).toISOString();
    const longRows = db
      .prepare(
        `SELECT summary FROM sessions WHERE created_at > :cutoff AND summary IS NOT NULL`,
      )
      .all({ cutoff: longCutoff });

    const recentSummaries = longRows.map((row) => row.summary ?? '');

    return { agentSessionCounts, driftSessions, recentSummaries };
  } catch {
    return emptyResult;
  } finally {
    db.close();
  }
}

// ---------------------------------------------------------------------------
// Flow registry helpers
// ---------------------------------------------------------------------------

/**
 * Load all registered flow IDs from `.github/flows/*.flow.yml`.
 * @returns {Promise<string[]>} Sorted list of flow IDs.
 */
async function loadFlowIds() {
  const flowsAbsDir = path.join(repoRoot, FLOWS_DIR);
  let entries;
  try {
    entries = await readdir(flowsAbsDir);
  } catch {
    return [];
  }

  const flowIds = [];
  for (const filename of entries) {
    if (!filename.endsWith('.flow.yml') || filename === 'flow.schema.yml') continue;
    // Derive flow ID from filename: e.g., `04.scoped-fix.flow.yml` → `04.scoped-fix`
    flowIds.push(filename.replace(/\.flow\.yml$/, ''));
  }

  return flowIds.toSorted();
}

/**
 * Compute underused flows by cross-referencing all flow IDs against session summaries.
 * A flow is considered underused if it is absent from all provided session summaries.
 * @param {string[]} allFlowIds - All registered flow IDs.
 * @param {string[]} recentSummaries - Session summary texts for the detection window.
 * @returns {{ flowId: string, mentionCount: number }[]}
 */
function computeUnderusedFlows(allFlowIds, recentSummaries) {
  return allFlowIds.map((flowId) => {
    const mentionCount = recentSummaries.filter((summary) => summary.includes(flowId)).length;
    return { flowId, mentionCount };
  });
}

// ---------------------------------------------------------------------------
// Recommendation builder
// ---------------------------------------------------------------------------

/**
 * Build recommended actions from aggregation results.
 * @param {{ gateFailureFrequency: object[], escalationCount: number, runtimeEnforcementEvidence: object, agentDriftSessions: object[], underusedFlows: object[] }} data
 * @returns {string[]}
 */
function buildRecommendedActions({
  gateFailureFrequency,
  escalationCount,
  runtimeEnforcementEvidence,
  agentDriftSessions,
  underusedFlows,
}) {
  const actions = [];

  if (
    gateFailureFrequency.length === 0 &&
    escalationCount === 0 &&
    runtimeEnforcementEvidence.proofMismatches === 0 &&
    runtimeEnforcementEvidence.missingPostActionPairs.length === 0 &&
    runtimeEnforcementEvidence.postWithoutPrePairs.length === 0
  ) {
    actions.push('No real gate failures or escalations detected in this window — gate health is good.');
  }

  for (const entry of gateFailureFrequency.slice(0, 3)) {
    actions.push(
      `Gate '${entry.gateId}' failed ${entry.failureCount} time(s) affecting: ${entry.affectedAgents.join(', ')}. Investigate root cause.`,
    );
  }

  if (escalationCount > 0) {
    actions.push(
      `${escalationCount} escalation(s) to 00-helping triggered. Review learning log for consecutive gate failure clusters.`,
    );
  }

  if (runtimeEnforcementEvidence.proofMismatches > 0) {
    actions.push(
      `${runtimeEnforcementEvidence.proofMismatches} runtime proof mismatch event(s) were recorded. Inspect the repo-owned context carrier or the delegator-chain preparation path.`,
    );
  }

  if (runtimeEnforcementEvidence.missingPostActionPairs.length > 0) {
    actions.push(
      `${runtimeEnforcementEvidence.missingPostActionPairs.length} action(s) logged a pre-action pass without a matching post-action pass. Inspect PostToolUse coverage or blocked follow-through.`,
    );
  }

  if (runtimeEnforcementEvidence.postWithoutPrePairs.length > 0) {
    actions.push(
      `${runtimeEnforcementEvidence.postWithoutPrePairs.length} action(s) logged a post-action pass without a matching pre-action pass. Inspect for bypassed pretool enforcement.`,
    );
  }

  if (agentDriftSessions.length > 0) {
    actions.push(
      `${agentDriftSessions.length} session(s) show no named flow ID in summary — consider promoting flow adoption.`,
    );
  }

  const zeroMentionFlows = underusedFlows.filter((flow) => flow.mentionCount === 0);
  if (zeroMentionFlows.length > 0 && underusedFlows.length > 0) {
    const sample = zeroMentionFlows
      .slice(0, 5)
      .map((flow) => flow.flowId)
      .join(', ');
    actions.push(
      `${zeroMentionFlows.length} flow(s) have zero mention in recent session summaries: ${sample}. ` +
        'These may be underused or newly added without session coverage yet.',
    );
  }

  return actions;
}

// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------

/**
 * Print a human-readable version of the audit report.
 * @param {object} report - Structured audit report.
 */
function printHumanReport(report) {
  console.log(`\nWorkflow Gap Audit — ${report.reportDate}`);
  console.log(`Window: ${report.windowDays} day(s)\n`);
  console.log(`Gate failures (real, filtered):  ${report.gateFailureFrequency.length} gate(s) with failures`);
  console.log(`Escalation count:                ${report.escalationCount}`);
  console.log(`Runtime proof mismatches:        ${report.runtimeEnforcementEvidence.proofMismatches}`);
  console.log(`Missing post-action pairs:       ${report.runtimeEnforcementEvidence.missingPostActionPairs.length}`);
  console.log(`Agent drift sessions:            ${report.agentDriftSessions.length}`);
  console.log(`Top failing gate:                ${report.topFailingGate ?? '(none)'}`);
  console.log(`\nRecommended actions:`);
  for (const action of report.recommendedActions) {
    console.log(`  - ${action}`);
  }
}

/**
 * Print usage text for the workflow-gap-audit script.
 */
function printUsage() {
  console.log(`workflow-gap-audit — Learning-event analytics aggregator

Usage:
  node scripts/agent-customization/workflow-gap-audit.mjs [options]

Options:
  --json             Emit structured JSON report to stdout.
  --window=<days>    Rolling window in days for aggregations (default: 7).
  --db=<path>        Absolute path to the SQLite session store file for session aggregations.
  --help, -h         Show this help text.

Examples:
  node scripts/agent-customization/workflow-gap-audit.mjs --json --window=14
  node scripts/agent-customization/workflow-gap-audit.mjs --json --db=/path/to/session.db

Notes:
  - Gate exceptions with sessionId "test-session-001" are filtered as test artifacts.
  - Session store aggregations (agentSessionCounts, agentDriftSessions, underusedFlows)
    require --db=<path>; without it, those fields return empty arrays.
`);
}
