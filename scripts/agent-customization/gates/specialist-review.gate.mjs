#!/usr/bin/env node
/**
 * Tier-1 gate: specialist-review
 *
 * Validates that the orchestrator dispatched specialist review agents
 * (Tier-3 scouts/specialists) to review 04-implementing output BEFORE
 * dispatching 05-green-testing. This gate checks the active plan's
 * validation-evidence section for evidence of specialist review.
 *
 * TWO EVIDENCE ANCHOR FORMS ARE ACCEPTED:
 *
 * 1. HEADING (canonical) — a Markdown heading whose text names a
 *    validation-evidence section: `## Latest validation evidence`,
 *    `### Validation evidence`, `#### FINAL VALIDATION EVIDENCE`. This is
 *    the form live plans carry, and the form the sibling gates
 *    (`plan-readiness`, `step-packet`, `convergence-tracker`) already read.
 * 2. TOKEN (legacy) — the underscore token `VALIDATION_EVIDENCE` /
 *    `validation_evidence:` used by older plans and step-packet YAML blocks.
 *
 * Gate contract: { pass: boolean, evidence: object, fixHint: string, owner: string }
 *
 * Usage:
 *   node scripts/agent-customization/gates/specialist-review.gate.mjs [--json] [--plan=path]
 */

import { readFileSync, readdirSync } from 'node:fs';
import path from 'node:path';
import { parseArgs } from '../customization-utils.mjs';

const REPO_ROOT = path.resolve(
  path.dirname(new URL(import.meta.url).pathname).replace(/^\/([A-Z]:)/, '$1'),
  '..',
  '..',
  '..',
);

const options = parseArgs(process.argv.slice(2));

const planDir = path.join(REPO_ROOT, 'plans');

/**
 * Read the EXPLICIT `--plan` value from the raw argv.
 *
 * `parseArgs` injects a DEFAULT plan
 * (`plans/completed/Agentic_Workflow_Architecture.plans.md`) when `--plan`
 * is absent. That default is a COMPLETED plan, so trusting it would
 * silently validate a stale artifact instead of the live plan and mask a
 * real RED. The gate therefore resolves the plan from the raw argv and
 * falls back to the active-plan scan when `--plan` was not passed.
 *
 * @param {string[]} argv - Raw CLI arguments.
 * @returns {string|null} The explicit plan path, or null when absent.
 */
function readExplicitPlanArg(argv) {
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg.startsWith('--plan=')) return arg.slice('--plan='.length).trim();
    if (arg === '--plan' && argv[index + 1] !== undefined) {
      return argv[index + 1].trim();
    }
  }
  return null;
}

const explicitPlanArg = readExplicitPlanArg(process.argv.slice(2));
const planArg = explicitPlanArg
  ? path.resolve(REPO_ROOT, explicitPlanArg)
  : null;

function findActivePlans() {
  if (planArg) {
    return [planArg];
  }
  try {
    return readdirSync(planDir)
      .filter((f) => f.endsWith('.plans.md'))
      .map((f) => path.join(planDir, f));
  } catch {
    return [];
  }
}

// --- Validation-evidence region discovery --------------------------------

/** Matches a Markdown ATX heading and captures its level and text. */
const HEADING_RE = /^(#{1,6})[ \t]+(.*)$/;

/**
 * True when a heading's text names a validation-evidence section.
 *
 * Accepts the canonical SPACED form (`Latest validation evidence`) and the
 * legacy UNDERSCORE form (`VALIDATION_EVIDENCE`).
 *
 * @param {string} text - Heading text.
 * @returns {boolean} Whether the heading names a validation-evidence section.
 */
function isEvidenceHeading(text) {
  return (
    /validation[ \t]+evidence/i.test(text) || /VALIDATION_EVIDENCE/.test(text)
  );
}

/**
 * Collect every validation-evidence region in the plan.
 *
 * Heading anchors are resolved OUTSIDE fenced code blocks so a heading
 * quoted inside a ``` fence is not mistaken for a live section. A
 * heading-anchored region runs from the heading to the next heading of the
 * SAME OR SHALLOWER level, so `###` sub-blocks stay inside their `##`
 * section. A token-anchored region runs from the token to the next `##` /
 * `PlanUpdate` / end of file.
 *
 * @param {string} content - Raw plan Markdown text.
 * @returns {string[]} Evidence region bodies (may contain empty strings).
 */
function collectEvidenceRegions(content) {
  const regions = [];

  // (1) Legacy token anchors — `VALIDATION_EVIDENCE` / `validation_evidence:`.
  for (const match of content.matchAll(
    /VALIDATION_EVIDENCE[\s\S]*?(?=##|PlanUpdate|$)/gi,
  )) {
    regions.push(match[0]);
  }

  // (2) Canonical heading anchors — `## Latest validation evidence`.
  const lines = content.split(/\r?\n/);
  const headings = [];
  let inCodeFence = false;

  for (let lineIndex = 0; lineIndex < lines.length; lineIndex += 1) {
    const trimmed = lines[lineIndex].trim();
    if (trimmed.startsWith('```')) {
      inCodeFence = !inCodeFence;
      continue;
    }
    if (inCodeFence) continue;
    const match = HEADING_RE.exec(trimmed);
    if (!match) continue;
    headings.push({ level: match[1].length, text: match[2], lineIndex });
  }

  for (let index = 0; index < headings.length; index += 1) {
    const anchor = headings[index];
    if (!isEvidenceHeading(anchor.text)) continue;
    let endLine = lines.length;
    for (let next = index + 1; next < headings.length; next += 1) {
      if (headings[next].level <= anchor.level) {
        endLine = headings[next].lineIndex;
        break;
      }
    }
    regions.push(lines.slice(anchor.lineIndex + 1, endLine).join('\n'));
  }

  return regions;
}

function checkPlanForSpecialistReview(planPath) {
  let content;
  try {
    content = readFileSync(planPath, 'utf8');
  } catch {
    return {
      plan: path.basename(planPath),
      found: false,
      reason: 'file not readable',
    };
  }

  if (!content.includes('[WIP]')) {
    return {
      plan: path.basename(planPath),
      found: false,
      reason: 'no [WIP] sections',
    };
  }

  // Look for specialist review evidence markers in VALIDATION_EVIDENCE.
  // TRIVIAL slices are exempt — the gate passes if the section contains
  // a TRIVIAL classification marker OR at least 1 specialist marker.
  const specialistMarkers = [
    /specialist[_\s-]review/i,
    /implementation-pattern-scout/i,
    /nge-core-scout/i,
    /performance-trace-specialist/i,
    /coverage-scout/i,
    /nge-benchmark-scout/i,
    /browser-runtime-scout/i,
    /visualizer-scout/i,
    /APPROVE/i,
    /REQUEST_CHANGES/i,
  ];

  const trivialMarkers = [
    /severity:\s*TRIVIAL/i,
    /TRIVIAL.*skip/i,
    /specialistCount:\s*0/i,
  ];

  // Collect EVERY validation-evidence region of the plan, not just the
  // first one. A long multi-step plan carries one evidence block per step
  // packet and per PlanUpdate, so a leftmost-match scan reads the evidence
  // of a CLOSED step and reports "no specialist review evidence" for a step
  // that was in fact reviewed.
  const evidenceSections = collectEvidenceRegions(content).filter(
    (section) => section.trim().length > 0,
  );

  if (evidenceSections.length === 0) {
    return {
      plan: path.basename(planPath),
      found: false,
      reason: 'no validation evidence section found',
    };
  }

  const evidenceText = evidenceSections.join('\n');
  const matchedSpecialist = specialistMarkers.filter((m) =>
    m.test(evidenceText),
  );
  const matchedTrivial = trivialMarkers.filter((m) => m.test(evidenceText));

  // Pass if TRIVIAL exemption marker found OR at least 1 specialist marker
  const found = matchedTrivial.length > 0 || matchedSpecialist.length >= 1;

  return {
    plan: path.basename(planPath),
    found,
    markers: matchedSpecialist.map((m) => m.source),
    trivialExempt: matchedTrivial.length > 0,
    reason: found
      ? matchedTrivial.length > 0
        ? 'TRIVIAL slice — specialist review exempt'
        : 'specialist review evidence found in the validation evidence section'
      : 'no specialist review evidence found',
  };
}

const activePlans = findActivePlans();
const wipPlans = activePlans.filter((p) => {
  try {
    return readFileSync(p, 'utf8').includes('[WIP]');
  } catch {
    return false;
  }
});

if (wipPlans.length === 0) {
  const noWipResult = {
    pass: true,
    evidence: {
      gate: 'specialist-review',
      tier: 1,
      check: 'No [WIP] plans found — specialist review gate not applicable.',
      wipPlans: 0,
    },
    fixHint:
      'No action needed. When a plan is [WIP] with FULL implementation slices, dispatch 1 Tier-3 specialist to review each 04-implementing slice before dispatching 05-green-testing. TRIVIAL slices (docs, comments, formatting) skip specialist review.',
    owner: 'orchestrator (Agent Zero)',
  };
  if (options.json) {
    console.log(JSON.stringify(noWipResult, null, 2));
  } else {
    console.log('PASS', 'specialist-review gate — no [WIP] plans');
  }
  process.exitCode = 0;
} else {
  const results = wipPlans.map(checkPlanForSpecialistReview);
  const allFound = results.every((r) => r.found);

  const result = {
    pass: allFound,
    evidence: {
      gate: 'specialist-review',
      tier: 1,
      check:
        'Each [WIP] plan with implementation slices must have specialist review evidence in its validation evidence section (## Latest validation evidence) before 05-green-testing is dispatched.',
      wipPlans: wipPlans.length,
      planResults: results,
    },
    fixHint: allFound
      ? 'Specialist review evidence confirmed.'
      : "Dispatch 1 Tier-3 specialist (e.g., implementation-pattern-scout or the domain-aligned scout) to review each FULL 04-implementing slice BEFORE dispatching 05-green-testing. TRIVIAL slices (docs, comments, formatting, plan-only) skip specialist review — record 'severity: TRIVIAL' in the plan's validation evidence section. Record APPROVE/REQUEST_CHANGES verdicts in the plan's `## Latest validation evidence` section.",
    owner: 'orchestrator (Agent Zero)',
  };

  if (options.json) {
    console.log(JSON.stringify(result, null, 2));
  } else {
    console.log(result.pass ? 'PASS' : 'FAIL', 'specialist-review gate');
    if (!result.pass) {
      for (const r of results) {
        if (!r.found) {
          console.log(`  ${r.plan}: ${r.reason}`);
        }
      }
      console.log('fixHint:', result.fixHint);
    }
  }
  process.exitCode = result.pass ? 0 : 1;
}
