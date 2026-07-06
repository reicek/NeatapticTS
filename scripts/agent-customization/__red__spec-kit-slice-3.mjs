#!/usr/bin/env node
/**
 * RED-phase check for Spec-Kit Assimilation Slice 3.
 *
 * Asserts AC-010..AC-013 for traceability IDs (AC-###) and coverage table.
 * This script must FAIL before implementation and PASS after.
 */

import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const root = join(__dirname, '..', '..');

const acceptancePath = join(
  root,
  '.github',
  'skills',
  'planning-acceptance-criteria',
  'SKILL.md',
);
const handoffPath = join(
  root,
  '.github',
  'skills',
  'phase-handoff-workflow',
  'SKILL.md',
);
const agentPath = join(root, '.github', 'agents', '01-planning.agent.md');

const acceptance = await readFile(acceptancePath, 'utf8');
const handoff = await readFile(handoffPath, 'utf8');
const agent = await readFile(agentPath, 'utf8');

const checks = [];

// AC-010: planning-acceptance-criteria/SKILL.md requires optional id: AC-###.
const ac010Rule =
  /id\s*[:=]\s*AC-\d{3,}/i.test(acceptance) &&
  /optional\s+id|id\s+(?:field\s+)?is\s+optional|optional\s+`id`|optionally\s+give.*acceptance\s+criteri(?:on|a).*id/i.test(
    acceptance,
  ) &&
  /acceptance\s+criteri(?:on|a).*id|id\s+field.*acceptance/i.test(acceptance);
checks.push({
  id: 'AC-010',
  desc: 'planning-acceptance-criteria/SKILL.md requires optional id: AC-### on criteria',
  pass: ac010Rule,
});

// AC-011: Before/After examples demonstrate AC-### IDs.
const beforeAfterSection = acceptance.indexOf('## Before/After Examples');
const sectionText =
  beforeAfterSection !== -1 ? acceptance.slice(beforeAfterSection) : '';
const beforeMatch = sectionText.match(
  /\*\*Before[^*]*\*\*[\s\S]*?(?=\*\*After)/i,
);
const afterMatch = sectionText.match(/\*\*After[^*]*\*\*[\s\S]*/i);
const beforeExampleText = beforeMatch ? beforeMatch[0] : '';
const afterExampleText = afterMatch ? afterMatch[0] : '';
const ac011Before =
  beforeExampleText.length > 0 &&
  !/id\s*[:=]\s*AC-\d{3,}/i.test(beforeExampleText);
const ac011After = /id\s*[:=]\s*AC-\d{3,}/i.test(afterExampleText);
checks.push({
  id: 'AC-011',
  desc: 'planning-acceptance-criteria/SKILL.md Before/After examples show AC-### IDs',
  pass: ac011Before && ac011After,
  detail: { hasBeforeWithoutId: ac011Before, hasAfterWithId: ac011After },
});

// AC-012: phase-handoff-workflow/SKILL.md documents a traceability table.
const ac012Table =
  /traceability/i.test(handoff) &&
  /AC-\d{3,}/i.test(handoff) &&
  /files?_changed|files changed/i.test(handoff) &&
  /validation\s+command|validation.*command/i.test(handoff);
checks.push({
  id: 'AC-012',
  desc: 'phase-handoff-workflow/SKILL.md documents traceability table mapping AC-### to files changed and validation command',
  pass: ac012Table,
});

// AC-013: 01-planning.agent.md Plan Block Schema examples show acceptance criteria with id fields.
const planBlockSection = agent.indexOf('## Plan Block Schemas');
const planBlockText =
  planBlockSection !== -1 ? agent.slice(planBlockSection) : '';
const ac013 =
  planBlockSection !== -1 &&
  /acceptance_criteria:/i.test(planBlockText) &&
  /id\s*[:=]\s*AC-\d{3,}/i.test(planBlockText);
checks.push({
  id: 'AC-013',
  desc: '01-planning.agent.md Plan Block Schema examples show acceptance criteria with id: AC-###',
  pass: ac013,
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
  console.log('\nAll AC-010..AC-013 checks pass.');
  process.exit(0);
} else {
  console.log('\nOne or more AC-010..AC-013 checks failed.');
  process.exit(1);
}
