#!/usr/bin/env node
/**
 * Tier-1 gate: agent-graph
 *
 * Wraps validate-agent-graph.mjs and transforms its structured JSON report into
 * the Tier-1 gate contract format. Checks that all agent delegation references
 * in .github/agents/ resolve to real files and that no delegation cycles exist.
 *
 * Gate contract: { pass: boolean, evidence: object, fixHint: string, owner: string }
 *
 * Usage:
 *   node scripts/agent-customization/gates/agent-graph.gate.mjs [--json]
 */

import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { parseArgs, repoRoot } from '../customization-utils.mjs';

const options = parseArgs(process.argv.slice(2));

const result = await runAgentGraphGate();

if (options.json) {
  console.log(JSON.stringify(result, null, 2));
} else {
  console.log(result.pass ? 'PASS' : 'FAIL', 'agent-graph gate');
  if (!result.pass) console.log('fixHint:', result.fixHint);
}

process.exitCode = result.pass ? 0 : 1;

// ---------------------------------------------------------------------------

async function runAgentGraphGate() {
  // Step 1: Invoke validate-agent-graph.mjs as a child process to get structured output.
  const scriptPath = path.join(
    repoRoot,
    'scripts',
    'agent-customization',
    'validate-agent-graph.mjs',
  );

  const spawned = spawnSync('node', [scriptPath, '--json'], {
    encoding: 'utf8',
    timeout: 15_000,
    cwd: repoRoot,
  });

  // Step 2: Parse the JSON report from the child process.
  let innerReport;
  try {
    innerReport = JSON.parse(spawned.stdout);
  } catch {
    return {
      pass: false,
      evidence: {
        stdout: spawned.stdout?.slice(0, 500) ?? '',
        stderr: spawned.stderr?.slice(0, 500) ?? '',
        exitCode: spawned.status,
      },
      fixHint:
        'validate-agent-graph.mjs did not return valid JSON. Run it directly to diagnose the failure.',
      owner: 'validate-agent-graph.mjs',
    };
  }

  // Step 3: Transform the inner report into the gate contract format.
  return {
    pass: innerReport.ok === true,
    evidence: {
      ok: innerReport.ok,
      issueCount: (innerReport.issues ?? []).length,
      agentCount: (innerReport.graph ?? []).length,
      issues: innerReport.issues ?? [],
    },
    fixHint: innerReport.ok
      ? 'Agent delegation graph is valid; all references resolve and no cycles detected.'
      : `Fix agent graph issues: ${(innerReport.issues ?? []).map((issue) => issue.message).join('; ')}`,
    owner: 'validate-agent-graph.mjs',
  };
}
