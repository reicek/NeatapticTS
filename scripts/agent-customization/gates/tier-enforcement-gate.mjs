#!/usr/bin/env node
/**
 * Tier-1 gate: tier-enforcement
 *
 * Runs validate-agent-graph.mjs and returns a tier-focused gate contract that
 * summarizes whether all agents have valid tier assignments and legal
 * delegation edges.
 *
 * Gate contract: { pass: boolean, evidence: object, fixHint: string, owner: string }
 *
 * Usage:
 *   node scripts/agent-customization/gates/tier-enforcement-gate.mjs [--json]
 */

import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { parseArgs } from '../customization-utils.mjs';
import { runValidateAgentGraph } from '../validate-agent-graph.mjs';

const options = parseArgs(process.argv.slice(2));

export async function runTierEnforcementGate({
  workspaceRoot = process.cwd(),
} = {}) {
  const validationReport = await runValidateAgentGraph({ workspaceRoot });
  const tierIssues = (validationReport.issues ?? []).filter((currentIssue) =>
    /tier|user-invocable/i.test(currentIssue.message),
  );

  return {
    pass: tierIssues.length === 0,
    evidence: {
      ok: validationReport.ok,
      issueCount: tierIssues.length,
      byTier: validationReport.inventory?.summary?.by_tier ?? null,
      userInvocableTotal:
        validationReport.inventory?.summary?.user_invocable_total ?? null,
      issues: tierIssues,
    },
    fixHint:
      tierIssues.length === 0
        ? 'Tier metadata is consistent with the delegation graph policy.'
        : `Fix tier metadata or delegation edges: ${tierIssues.map((currentIssue) => currentIssue.message).join('; ')}`,
    owner: 'validate-agent-graph.mjs',
  };
}

async function main() {
  const report = await runTierEnforcementGate();

  if (options.json) {
    console.log(JSON.stringify(report, null, 2));
  } else {
    console.log(report.pass ? 'PASS' : 'FAIL', 'tier-enforcement gate');
    if (!report.pass) console.log('fixHint:', report.fixHint);
  }

  process.exitCode = report.pass ? 0 : 1;
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href
) {
  await main();
}
