#!/usr/bin/env node
/**
 * RED-phase check for Spec-Kit Assimilation Slice 10.
 *
 * Asserts AC-036..AC-039 for cross-cutting Spec-Kit phrasing and Tier-1 agent polish.
 * This script must FAIL before implementation and PASS after.
 *
 * Audit logic:
 *  - AC-036: every Tier-1 agent file must contain at least one of the four exact
 *    Spec-Kit phrases: "unit tests for English", "append-only convergence",
 *    "gap types: missing/partial/contradicts/unrequested", or "constitution authority".
 *    We scan the whole agent file because the phrases may naturally live in the
 *    description, purpose, constraints, or Default Flow sections.
 *  - AC-037: the three target skills must each contain at least one of the four phrases.
 *  - AC-038: all eight Tier-1 agents must share consistent frontmatter:
 *    tier: 1, user-invocable: true, disable-model-invocation: false, and a
 *    non-empty model string. We read the frontmatter block between the first
 *    `---` pair and assert these fields. The repo currently uses kebab-case keys.
 *  - AC-039: every Tier-1 agent must contain at least one `Reference:` citation
 *    line pointing to a skill, OR cleanly delegate to skills without inlining
 *    durable rules. To keep the check conservative and machine-verifiable, we
 *    require at least one `Reference:` line per agent. Agents that previously
 *    inlined durable workflow rules now cite the canonical skill instead.
 */

import { readFileSync, existsSync } from 'node:fs';

const checks = [];

function check(id, condition, message) {
  if (!condition) {
    checks.push({ id, pass: false, message });
    console.error(`FAIL ${id}: ${message}`);
  } else {
    checks.push({ id, pass: true, message: 'ok' });
    console.log(`PASS ${id}`);
  }
}

const agents = [
  '.github/agents/00-helping.agent.md',
  '.github/agents/01-planning.agent.md',
  '.github/agents/02-researching.agent.md',
  '.github/agents/03-red-testing.agent.md',
  '.github/agents/04-implementing.agent.md',
  '.github/agents/05-green-testing.agent.md',
  '.github/agents/06-documenting.agent.md',
  '.github/agents/07-logging.agent.md',
];

const skills = [
  '.github/skills/planning-acceptance-criteria/SKILL.md',
  '.github/skills/phase-handoff-workflow/SKILL.md',
  '.github/skills/plan-sync-validation/SKILL.md',
];

const phrases = [
  'unit tests for English',
  'append-only convergence',
  'gap types: missing/partial/contradicts/unrequested',
  'constitution authority',
];

function hasAnyPhrase(text) {
  return phrases.some((phrase) => text.includes(phrase));
}

function parseFrontmatter(text) {
  const match = text.match(/^---\n([\s\S]*?)\n---/);
  if (!match) return null;
  const raw = match[1];
  const result = {};
  for (const line of raw.split('\n')) {
    const colonIndex = line.indexOf(':');
    if (colonIndex === -1) continue;
    const key = line.slice(0, colonIndex).trim();
    let value = line.slice(colonIndex + 1).trim();
    // Strip leading/trailing quotes if present
    if (
      (value.startsWith("'") && value.endsWith("'")) ||
      (value.startsWith('"') && value.endsWith('"'))
    ) {
      value = value.slice(1, -1);
    }
    result[key] = value;
  }
  return result;
}

// AC-036: Spec-Kit phrase coverage in every Tier-1 agent.
for (let i = 0; i < agents.length; i++) {
  const agentPath = agents[i];
  const exists = existsSync(agentPath);
  const text = exists ? readFileSync(agentPath, 'utf-8') : '';
  const ok = exists && hasAnyPhrase(text);
  check(
    `AC-036-${String(i + 1).padStart(2, '0')}`,
    ok,
    exists
      ? `${agentPath} missing Spec-Kit phrase (one of: ${phrases.join('; ')}).`
      : `${agentPath} does not exist`,
  );
}

// AC-037: Spec-Kit phrase coverage in the three target skills.
for (let i = 0; i < skills.length; i++) {
  const skillPath = skills[i];
  const exists = existsSync(skillPath);
  const text = exists ? readFileSync(skillPath, 'utf-8') : '';
  const ok = exists && hasAnyPhrase(text);
  check(
    `AC-037-${String(i + 1).padStart(2, '0')}`,
    ok,
    exists
      ? `${skillPath} missing Spec-Kit phrase (one of: ${phrases.join('; ')}).`
      : `${skillPath} does not exist`,
  );
}

// AC-038: consistent Tier-1 frontmatter.
for (let i = 0; i < agents.length; i++) {
  const agentPath = agents[i];
  const exists = existsSync(agentPath);
  if (!exists) {
    check(
      `AC-038-${String(i + 1).padStart(2, '0')}`,
      false,
      `${agentPath} does not exist`,
    );
    continue;
  }
  const text = readFileSync(agentPath, 'utf-8');
  const fm = parseFrontmatter(text);
  const tierOk = fm && String(fm.tier) === '1';
  const userInvocableOk = fm && String(fm['user-invocable']) === 'true';
  const disableModelInvocationOk =
    fm && String(fm['disable-model-invocation']) === 'false';
  const modelOk =
    fm && typeof fm.model === 'string' && fm.model.trim().length > 0;
  const allOk =
    tierOk && userInvocableOk && disableModelInvocationOk && modelOk;
  check(
    `AC-038-${String(i + 1).padStart(2, '0')}`,
    allOk,
    `${agentPath} frontmatter inconsistent: tier=${fm?.tier}, user-invocable=${fm?.['user-invocable']}, disable-model-invocation=${fm?.['disable-model-invocation']}, model=${fm?.model}`,
  );
}

// AC-039: every Tier-1 agent has at least one Reference: citation.
for (let i = 0; i < agents.length; i++) {
  const agentPath = agents[i];
  const exists = existsSync(agentPath);
  if (!exists) {
    check(
      `AC-039-${String(i + 1).padStart(2, '0')}`,
      false,
      `${agentPath} does not exist`,
    );
    continue;
  }
  const text = readFileSync(agentPath, 'utf-8');
  const hasReference = /^Reference:.+$/m.test(text);
  check(
    `AC-039-${String(i + 1).padStart(2, '0')}`,
    hasReference,
    `${agentPath} missing a "Reference:" citation to a canonical skill.`,
  );
}

const failed = checks.filter((c) => !c.pass);
if (failed.length > 0) {
  console.error(
    `\nRED check failed: ${failed.length}/${checks.length} AC(s) not satisfied.`,
  );
  process.exit(1);
}

console.log(
  `\nGREEN check passed: ${checks.length}/${checks.length} AC(s) satisfied.`,
);
process.exit(0);
