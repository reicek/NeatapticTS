import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const root = join(__dirname, '..', '..');

const agentPath = join(root, '.github', 'agents', '01-planning.agent.md');
const flowPath = join(root, '.github', 'flows', '01.phase-kickoff.flow.yml');

const agent = await readFile(agentPath, 'utf8');
const flow = await readFile(flowPath, 'utf8');

const checks = [];

// AC-006: agent documents the ≤3 NEEDS CLARIFICATION marker cap.
const ac006 =
  /NEEDS CLARIFICATION/i.test(agent) &&
  /(?:no more than|at most|≤|<=)\s*3\s+(?:NEEDS CLARIFICATION|markers|ambiguities)/i.test(
    agent,
  );
checks.push({
  id: 'AC-006',
  desc: '01-planning.agent.md caps NEEDS CLARIFICATION markers at ≤3',
  pass: ac006,
});

// AC-007: agent documents the ≤5 focused-question cap.
const ac007 =
  /at most\s+5/i.test(agent) &&
  /(?:focused\s+questions|clarification\s+questions|questions)/i.test(agent);
checks.push({
  id: 'AC-007',
  desc: '01-planning.agent.md asks at most 5 focused questions',
  pass: ac007,
});

// AC-008: agent describes an append-only ## Clarifications section.
const ac008 = /## Clarifications/i.test(agent) && /append-only/i.test(agent);
checks.push({
  id: 'AC-008',
  desc: '01-planning.agent.md describes append-only ## Clarifications section',
  pass: ac008,
});

// AC-009: flow references both the cap and the five-question limit in entry/exit checks.
const flowEntryExit = /entry-checks|exit-checks/i.test(flow);
const ac009Cap =
  flowEntryExit &&
  /NEEDS CLARIFICATION/i.test(flow) &&
  /(?:no more than|at most|≤|<=)\s*3/i.test(flow);
const ac009Limit =
  flowEntryExit &&
  /(?:at most|≤|<=)\s*5/i.test(flow) &&
  /(?:question|clarification)/i.test(flow);
const ac009 = ac009Cap && ac009Limit;
checks.push({
  id: 'AC-009',
  desc: '01.phase-kickoff.flow.yml references ≤3 NEEDS CLARIFICATION cap and ≤5 question limit in entry/exit checks',
  pass: ac009,
  detail: {
    hasEntryExit: flowEntryExit,
    hasCap: ac009Cap,
    hasLimit: ac009Limit,
  },
});

let allPass = true;
for (const c of checks) {
  const status = c.pass ? 'PASS' : 'FAIL';
  if (!c.pass) allPass = false;
  if (c.detail) {
    console.log(`${c.id}: ${status} — ${c.desc}`);
    console.log(`  detail: ${JSON.stringify(c.detail)}`);
  } else {
    console.log(`${c.id}: ${status} — ${c.desc}`);
  }
}

if (allPass) {
  console.log('\nAll AC-006..AC-009 checks pass.');
  process.exit(0);
} else {
  console.log('\nOne or more AC-006..AC-009 checks failed.');
  process.exit(1);
}
