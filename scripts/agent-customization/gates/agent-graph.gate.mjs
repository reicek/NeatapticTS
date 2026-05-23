#!/usr/bin/env node
/**
 * Tier-1 gate: agent-graph
 *
 * Runs validate-agent-graph.mjs as an importable helper and transforms its
 * structured report into the Tier-1 gate contract format. Checks that agent
 * delegation references resolve, no cycles exist, and tier enforcement rules hold.
 *
 * Gate contract: { pass: boolean, evidence: object, fixHint: string, owner: string }
 *
 * Usage:
 *   node scripts/agent-customization/gates/agent-graph.gate.mjs [--json]
 */

import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { parseArgs } from '../customization-utils.mjs';
import { runValidateAgentGraph } from '../validate-agent-graph.mjs';

const options = parseArgs(process.argv.slice(2));

export async function runAgentGraphGate({ workspaceRoot = process.cwd() } = {}) {
  const innerReport = await runValidateAgentGraph({ workspaceRoot });

  return {
    pass: innerReport.ok === true,
    evidence: {
      ok: innerReport.ok,
      issueCount: (innerReport.issues ?? []).length,
      agentCount: (innerReport.graph ?? []).length,
      byTier: innerReport.inventory?.summary?.by_tier ?? null,
      issues: innerReport.issues ?? [],
    },
    fixHint: innerReport.ok
      ? 'Agent delegation graph is valid; references resolve, no cycles exist, and tier enforcement rules pass.'
      : `Fix agent graph issues: ${(innerReport.issues ?? []).map((currentIssue) => currentIssue.message).join('; ')}`,
    owner: 'validate-agent-graph.mjs',
  };
}

async function main() {
  const result = await runAgentGraphGate();

  if (options.json) {
    console.log(JSON.stringify(result, null, 2));
  } else {
    console.log(result.pass ? 'PASS' : 'FAIL', 'agent-graph gate');
    if (!result.pass) console.log('fixHint:', result.fixHint);
  }

  process.exitCode = result.pass ? 0 : 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  await main();
}
