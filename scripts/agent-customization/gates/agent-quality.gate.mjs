#!/usr/bin/env node
/**
 * Tier-1 gate: agent-quality
 *
 * Runs validate-agent-quality.mjs and returns a gate contract that summarizes
 * whether every `.agent.md` file satisfies the agent quality contract.
 *
 * Gate contract: { pass: boolean, evidence: object, fixHint: string, owner: string }
 *
 * Usage:
 *   node scripts/agent-customization/gates/agent-quality.gate.mjs [--json]
 */

import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { parseArgs } from '../customization-utils.mjs';
import { runValidateAgentQuality } from '../validate-agent-quality.mjs';

const options = parseArgs(process.argv.slice(2));

export async function runAgentQualityGate() {
  const validationReport = await runValidateAgentQuality();
  const failingAgents = validationReport.agents.filter(
    (agentReport) => agentReport.counts.errors > 0,
  );

  return {
    pass: validationReport.ok,
    evidence: {
      contractDocument: validationReport.contractDocument,
      issueCount: validationReport.issues.length,
      counts: validationReport.counts,
      failingAgents: failingAgents.map((agentReport) => ({
        path: agentReport.path,
        name: agentReport.name,
        tier: agentReport.tier,
        errors: agentReport.counts.errors,
        warnings: agentReport.counts.warnings,
      })),
    },
    fixHint: validationReport.ok
      ? 'All agents satisfy the documented agent quality contract.'
      : 'Fix the missing agent sections or structured-v1 field mismatches reported by validate-agent-quality.mjs.',
    owner: 'validate-agent-quality.mjs',
  };
}

async function main() {
  const report = await runAgentQualityGate();

  if (options.json) {
    console.log(JSON.stringify(report, null, 2));
  } else {
    console.log(report.pass ? 'PASS' : 'FAIL', 'agent-quality gate');
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
