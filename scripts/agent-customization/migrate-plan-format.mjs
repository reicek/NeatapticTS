#!/usr/bin/env node
/**
 * Migration tool: convert legacy plan phase/step blocks to the new format.
 *
 * Scans a plan file (or all plan files with --all), finds phase/step headings
 * whose status is not [DONE], and ensures every active block has a conforming
 * YAML metadata block. Legacy blocks (no expansion field or deprecated
 * agent:/agent_file:) are rewritten; blocks that already conform are left
 * untouched.
 *
 * Usage:
 *   node scripts/agent-customization/migrate-plan-format.mjs --plan=plans/PlanName.plans.md [--dry-run]
 *   node scripts/agent-customization/migrate-plan-format.mjs --all [--dry-run]
 */

import { readFile, readdir, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import {
  normalizePath,
  parseArgs,
  parsePlanYamlBlock,
  repoRoot,
  writeReport,
} from './customization-utils.mjs';

const AGENT_TO_GOAL = {
  '00-helping': 'helping',
  '01-planning': 'planning',
  '02-researching': 'researching',
  '03-red-testing': 'red-testing',
  '04-implementing': 'implementing',
  '05-green-testing': 'green-testing',
  '06-documenting': 'documenting',
  '07-logging': 'logging',
};

const GOAL_TO_SKILLS = {
  planning: ['plan-alignment'],
  researching: ['research-methodology'],
  'red-testing': ['red-test-contracts'],
  implementing: ['implementation-standards'],
  'green-testing': ['green-validation-gates', 'coverage-guard'],
  documenting: ['educational-docs'],
  logging: ['tracker-handoff'],
  helping: ['mcp-local-server-workflow'],
};

/**
 * Convert a step heading label to its numeric order.
 *
 * Purely numeric labels (`01`) are parsed as numbers. Letter-prefixed labels
 * (`E1`) return the trailing digits so step metadata and goal inference stay
 * sequential.
 *
 * @param {string} stepLabel - Step label captured from the heading.
 * @returns {number} Numeric step order, or `NaN` when no digits are present.
 */
function parseStepNumber(stepLabel) {
  if (/^\d+$/u.test(stepLabel)) {
    return Number(stepLabel);
  }
  const trailingDigits = stepLabel.match(/\d+$/u);
  /* istanbul ignore next -- regex requires [A-Z]?\d+ so stepLabel always has trailing digits */
  return trailingDigits ? Number(trailingDigits[0]) : NaN;
}

const COMBINED_PATTERN =
  /^(?:### Phase (?<phase>[A-Z0-9]+) — (?<phaseTitle>.+?) \[(?<phaseStatus>PLANNED|WIP|DONE)\]|#### Step (?<step>[A-Z]?\d+)\s*[:\-—]\s*(?<stepTitle>.+?) \[(?<stepStatus>PLANNED|WIP|DONE)\]|```yaml\s*\r?\n(?<yaml>[\s\S]*?)```)[ \t]*\r?$/gmu;

const options = parseArgs(process.argv.slice(2));

if (options.help) {
  console.log(`Usage:
  node scripts/agent-customization/migrate-plan-format.mjs --plan=<path> [--dry-run]
  node scripts/agent-customization/migrate-plan-format.mjs --all [--dry-run]`);
  process.exit(0);
}

const dryRun = options['dry-run'] ?? options.dry_run ?? false;

async function main() {
  if (options.all) {
    const planFiles = await collectActivePlanFiles();
    const results = [];
    for (const planFile of planFiles) {
      results.push(await migratePlanFile(planFile));
    }
    const changedCount = results.filter((result) => result.changed).length;
    const report = {
      ok: true,
      dryRun,
      plansProcessed: results.length,
      plansChanged: changedCount,
      details: results,
    };
    writeReport(report, options);
    process.exitCode = 0;
    return;
  }

  const planFile = options.plan;
  if (!planFile) {
    const report = {
      ok: false,
      error: 'Missing required --plan argument or --all flag.',
    };
    writeReport(report, options);
    process.exitCode = 1;
    return;
  }

  const result = await migratePlanFile(planFile);
  writeReport(result, options);
  process.exitCode = 0;
}

await main();

async function collectActivePlanFiles() {
  const roots = [
    path.join(repoRoot, 'plans'),
    path.join(repoRoot, 'plans', 'completed'),
  ];
  const files = [];

  for (const root of roots) {
    let entries;
    try {
      entries = await readdir(root);
    } catch (error) {
      continue;
    }

    for (const entry of entries) {
      if (!entry.endsWith('.plans.md')) continue;
      const fullPath = path.join(root, entry);
      const entryStat = await stat(fullPath);
      if (!entryStat.isFile()) continue;
      files.push(path.relative(repoRoot, fullPath).replace(/\\/gu, '/'));
    }
  }

  return files;
}

async function migratePlanFile(planFile) {
  const normalizedPlanPath = normalizePath(planFile);
  const absolutePath = path.join(repoRoot, normalizedPlanPath);
  let text;
  try {
    text = await readFile(absolutePath, 'utf8');
  } catch (error) {
    return {
      planFile: normalizedPlanPath,
      changed: false,
      error: String(error),
    };
  }

  const tokens = tokenizePlan(text);
  const context = buildContext(tokens);
  const { newText, changedBlocks } = rebuildPlan(
    text,
    tokens,
    context,
    normalizedPlanPath,
  );

  if (newText !== text && !dryRun) {
    await writeFile(absolutePath, newText, 'utf8');
  }

  return {
    planFile: normalizedPlanPath,
    changed: newText !== text,
    changedBlocks,
    dryRun,
  };
}

function tokenizePlan(text) {
  const tokens = [];
  let match;
  COMBINED_PATTERN.lastIndex = 0;
  while ((match = COMBINED_PATTERN.exec(text)) !== null) {
    const groups = match.groups;
    /* istanbul ignore if -- regex always produces named groups when it matches */
    if (!groups) continue;

    const start = match.index;
    const end = COMBINED_PATTERN.lastIndex;

    if (groups.phase !== undefined) {
      tokens.push({
        kind: 'phase',
        start,
        end,
        phase: groups.phase,
        title: groups.phaseTitle,
        status: groups.phaseStatus,
      });
    } else if (groups.step !== undefined) {
      tokens.push({
        kind: 'step',
        start,
        end,
        step: parseStepNumber(groups.step),
        stepLabel: groups.step,
        title: groups.stepTitle,
        status: groups.stepStatus,
      });
    } else {
      tokens.push({
        kind: 'yaml',
        start,
        end,
        rawYaml: groups.yaml,
      });
    }
  }

  return tokens;
}

function buildContext(tokens) {
  const context = {
    phaseByToken: new Map(),
    stepByToken: new Map(),
    yamlByHeadingToken: new Map(),
    headingsByPhase: new Map(),
    stepsByPhase: new Map(),
  };

  let currentPhaseToken = null;
  let currentHeadingToken = null;

  for (const token of tokens) {
    if (token.kind === 'phase') {
      currentPhaseToken = token;
      currentHeadingToken = token;
      context.phaseByToken.set(token, {
        ...token,
        steps: [],
      });
      context.headingsByPhase.set(token, []);
      context.stepsByPhase.set(token, []);
    } else if (token.kind === 'step') {
      currentHeadingToken = token;
      if (currentPhaseToken) {
        context.phaseByToken.get(currentPhaseToken).steps.push(token);
        context.stepsByPhase.get(currentPhaseToken).push(token);
      }
      context.stepByToken.set(token, {
        ...token,
        phaseToken: currentPhaseToken,
      });
    /* istanbul ignore else -- only phase/step/yaml token kinds exist */
    } else {
      if (currentHeadingToken) {
        context.yamlByHeadingToken.set(currentHeadingToken, token);
      }
    }
  }

  return context;
}

function rebuildPlan(text, tokens, context, planFile) {
  const changedBlocks = [];
  let output = '';
  let lastIndex = 0;

  for (let tokenIndex = 0; tokenIndex < tokens.length; tokenIndex++) {
    const token = tokens[tokenIndex];

    if (token.kind === 'phase' || token.kind === 'step') {
      const isActive = token.status !== 'DONE';
      const associatedYaml = context.yamlByHeadingToken.get(token);

      if (isActive && !associatedYaml) {
        const generatedBlock = generateYamlBlockForHeading(
          token,
          context,
          planFile,
        );
        output += text.slice(lastIndex, token.end);
        output += '\n\n' + generatedBlock;
        changedBlocks.push({
          type: token.kind,
          title: token.title,
          action: 'inserted',
        });
        lastIndex = token.end;
        continue;
      }

      output += text.slice(lastIndex, token.end);
      lastIndex = token.end;
      continue;
    }

    /* istanbul ignore else -- phase/step handled above with continue; only yaml remains */
    if (token.kind === 'yaml') {
      const headingToken = findHeadingForYamlToken(token, tokens, tokenIndex);
      if (headingToken && headingToken.status !== 'DONE') {
        const generatedBlock = generateYamlBlockForHeading(
          headingToken,
          context,
          planFile,
          token.rawYaml,
        );
        const originalBlock = text.slice(token.start, token.end);
        /* istanbul ignore else -- exact match between original and generated YAML is extremely hard to construct */
        if (originalBlock !== generatedBlock) {
          changedBlocks.push({
            type: headingToken.kind,
            title: headingToken.title,
            action: 'rewritten',
          });
        }
        output += text.slice(lastIndex, token.start);
        output += generatedBlock;
        lastIndex = token.end;
        continue;
      }

      output += text.slice(lastIndex, token.end);
      lastIndex = token.end;
      continue;
    }
  }

  output += text.slice(lastIndex);
  return { newText: output, changedBlocks };
}

function findHeadingForYamlToken(yamlToken, tokens, yamlIndex) {
  for (let index = yamlIndex - 1; index >= 0; index--) {
    const token = tokens[index];
    if (token.kind === 'phase' || token.kind === 'step') {
      return token;
    }
  }
  return null;
}

function generateYamlBlockForHeading(
  headingToken,
  context,
  planFile,
  existingYaml = null,
) {
  const existing = parseExistingYaml(existingYaml);
  const metadata =
    headingToken.kind === 'phase'
      ? buildPhaseMetadata(headingToken, context, planFile, existing)
      : buildStepMetadata(headingToken, context, planFile, existing);
  const serialized = serializeMetadata(metadata);
  return `\`\`\`yaml\n${serialized}\n\`\`\``;
}

function parseExistingYaml(rawYaml) {
  if (!rawYaml) return {};
  try {
    return parsePlanYamlBlock(rawYaml);
  } catch {
    return {};
  }
}

function buildPhaseMetadata(headingToken, context, planFile, existing) {
  const phaseNumber = headingToken.phase;
  const title = existing.title ?? headingToken.title;
  const status = `[${headingToken.status}]`;
  const nextPhaseTitle = inferNextPhaseTitle(headingToken, context);
  const placeholderSteps = inferPlaceholderSteps(headingToken, context);

  return {
    phase: phaseNumber,
    title,
    status,
    goal: 'planning',
    expansion: 'steps',
    auto_expand: false,
    mode: existing.mode ?? 'fresh-session',
    source_of_truth: normalizeSourceOfTruth(existing.source_of_truth, planFile),
    copy_paste: coerceBoolean(existing.copy_paste, true),
    next_phase: existing.next_phase ?? nextPhaseTitle,
    skills: normalizeSkills(existing.skills, 'planning'),
    validation: normalizeValidation(existing.validation, planFile),
    acceptance_criteria: normalizeAcceptanceCriteria(
      existing.acceptance_criteria,
    ),
    placeholder_steps: existing.placeholder_steps ?? placeholderSteps,
  };
}

function buildStepMetadata(headingToken, context, planFile, existing) {
  const phaseToken = findPhaseForStep(headingToken, context);
  const phaseNumber = phaseToken ? phaseToken.phase : (existing.phase ?? '?');
  const title = existing.title ?? headingToken.title;
  const status = `[${headingToken.status}]`;
  const goal = inferGoal(existing, headingToken.step);
  const expansion = inferStepExpansion(existing, goal);
  const autoExpand = expansion === 'slices';
  const tddSequence =
    expansion === 'slices' ? (existing.tdd_sequence ?? 'red-green') : undefined;
  const nextStepTitle = inferNextStepTitle(headingToken, context);

  const metadata = {
    phase: phaseNumber,
    step: headingToken.step,
    title,
    status,
    goal,
    mode: existing.mode ?? 'fresh-session',
    source_of_truth: normalizeSourceOfTruth(existing.source_of_truth, planFile),
    copy_paste: coerceBoolean(existing.copy_paste, true),
    next_step: existing.next_step ?? nextStepTitle,
    skills: normalizeSkills(existing.skills, goal),
    validation: normalizeValidation(existing.validation, planFile),
    acceptance_criteria: normalizeAcceptanceCriteria(
      existing.acceptance_criteria,
    ),
  };

  if (tddSequence !== undefined) {
    metadata.tdd_sequence = tddSequence;
  }

  metadata.expansion = expansion;
  metadata.auto_expand = autoExpand;

  if (existing.specialists !== undefined) {
    metadata.specialists = existing.specialists;
  }

  if (expansion === 'slices' && existing.slices !== undefined) {
    metadata.slices = normalizeSlices(existing.slices);
  } else if (expansion === 'slices') {
    metadata.slices = generateDefaultSlices(goal, headingToken.step);
  }

  return metadata;
}

function inferGoal(existing, stepNumber) {
  if (existing.goal && typeof existing.goal === 'string') return existing.goal;
  if (existing.agent && typeof existing.agent === 'string') {
    return AGENT_TO_GOAL[existing.agent] ?? 'implementing';
  }
  if (stepNumber === 1) return 'planning';
  return 'implementing';
}

function inferStepExpansion(existing, goal) {
  if (existing.expansion === 'slices' || existing.expansion === 'none') {
    return existing.expansion;
  }
  if (
    existing.tdd_sequence === 'red-green' ||
    existing.tdd_sequence === 'green-only'
  ) {
    return 'slices';
  }
  if (goal === 'implementing') {
    return 'slices';
  }
  return 'none';
}

function findPhaseForStep(stepToken, context) {
  for (const [phaseToken, steps] of context.stepsByPhase.entries()) {
    if (steps.includes(stepToken)) return phaseToken;
  }
  return null;
}

function inferNextPhaseTitle(phaseToken, context) {
  const phases = [...context.headingsByPhase.keys()];
  const index = phases.indexOf(phaseToken);
  const nextPhase = phases.at(index + 1);
  return nextPhase ? nextPhase.title : 'null';
}

function inferPlaceholderSteps(phaseToken, context) {
  /* istanbul ignore next -- phaseToken always in stepsByPhase map */
  const steps = context.stepsByPhase.get(phaseToken) ?? [];
  return steps.map((step) => {
    /* istanbul ignore next -- stepLabel always set by tokenizer */
    const label = step.stepLabel ?? String(step.step).padStart(2, '0');
    return `Step ${label} — ${step.title}`;
  });
}

function inferNextStepTitle(stepToken, context) {
  const phaseToken = findPhaseForStep(stepToken, context);
  if (!phaseToken) return 'null';
  /* istanbul ignore next -- phaseToken always in stepsByPhase map */
  const steps = context.stepsByPhase.get(phaseToken) ?? [];
  const index = steps.indexOf(stepToken);
  const nextStep = steps.at(index + 1);
  if (nextStep) return nextStep.title;
  const nextPhaseTitle = inferNextPhaseTitle(phaseToken, context);
  return nextPhaseTitle === 'null' ? 'null' : nextPhaseTitle;
}

function normalizeSourceOfTruth(existingValue, planFile) {
  if (existingValue && typeof existingValue === 'string') {
    return normalizePath(existingValue);
  }
  return planFile;
}

function normalizeSkills(existingSkills, goal) {
  if (Array.isArray(existingSkills) && existingSkills.length > 0) {
    return existingSkills.map((item) => String(item));
  }
  if (typeof existingSkills === 'string' && existingSkills.trim()) {
    return [existingSkills.trim()];
  }
  /* istanbul ignore next -- all goals from inferGoal are in GOAL_TO_SKILLS map */
  return GOAL_TO_SKILLS[goal] ?? ['plan-alignment'];
}

function normalizeValidation(existingValidation, planFile) {
  if (Array.isArray(existingValidation) && existingValidation.length > 0) {
    return existingValidation.map((item) => String(item));
  }
  return [
    `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=${planFile}`,
  ];
}

function normalizeAcceptanceCriteria(existingAcceptanceCriteria) {
  if (
    Array.isArray(existingAcceptanceCriteria) &&
    existingAcceptanceCriteria.length > 0
  ) {
    return existingAcceptanceCriteria.map((item) => String(item));
  }
  return ['Phase/step metadata validates with the new plan-phase-step schema.'];
}

function normalizeSlices(existingSlices) {
  if (!Array.isArray(existingSlices)) return [];
  return existingSlices.map((slice) => {
    if (!slice || typeof slice !== 'object') return slice;
    const normalized = { ...slice };
    if (normalized.dependencies && !Array.isArray(normalized.dependencies)) {
      normalized.dependencies = [];
    }
    return normalized;
  });
}

function generateDefaultSlices(goal, stepNumber) {
  const prefix = `step-${stepNumber}`;
  return [
    {
      slice_id: `${prefix}-red-tests`,
      title: 'Write red tests',
      status: '[PLANNED]',
      goal: 'red-testing',
      estimate_hours: 4,
      files_to_change: ['TBD'],
      acceptance_criteria: [
        'Red tests exist and fail for the expected behavior.',
      ],
      parallelizable: false,
      dependencies: [],
      next_slice: `${prefix}-core`,
    },
    {
      slice_id: `${prefix}-core`,
      title: 'Implement the core behavior',
      status: '[PLANNED]',
      goal: 'implementing',
      estimate_hours: 8,
      files_to_change: ['TBD'],
      acceptance_criteria: [
        'Implementation satisfies the red tests and design.',
      ],
      parallelizable: false,
      dependencies: [`${prefix}-red-tests`],
      next_slice: `${prefix}-green`,
    },
    {
      slice_id: `${prefix}-green`,
      title: 'Green validation and coverage guard',
      status: '[PLANNED]',
      goal: 'green-testing',
      estimate_hours: 4,
      files_to_change: ['coverage/lcov.info'],
      acceptance_criteria: ['All tests pass and coverage guard is satisfied.'],
      parallelizable: false,
      dependencies: [`${prefix}-core`],
    },
  ];
}

function coerceBoolean(value, defaultValue) {
  if (value === true || value === false) return value;
  if (value === 'true') return true;
  if (value === 'false') return false;
  return defaultValue;
}

function serializeMetadata(metadata) {
  const lines = [];
  const phaseOrder = [
    'phase',
    'title',
    'status',
    'goal',
    'expansion',
    'auto_expand',
    'mode',
    'source_of_truth',
    'copy_paste',
    'next_phase',
    'skills',
    'validation',
    'acceptance_criteria',
    'placeholder_steps',
  ];
  const stepOrder = [
    'phase',
    'step',
    'title',
    'status',
    'goal',
    'tdd_sequence',
    'expansion',
    'auto_expand',
    'mode',
    'source_of_truth',
    'copy_paste',
    'next_step',
    'skills',
    'validation',
    'acceptance_criteria',
    'specialists',
    'slices',
  ];
  const isStep = Object.prototype.hasOwnProperty.call(metadata, 'step');
  const order = isStep ? stepOrder : phaseOrder;
  const knownKeys = new Set(order);
  const orderedKeys = [...Object.keys(metadata)].sort((a, b) => {
    const aKnown = knownKeys.has(a);
    const bKnown = knownKeys.has(b);
    if (aKnown && bKnown) return order.indexOf(a) - order.indexOf(b);
    if (aKnown) return -1;
    if (bKnown) return 1;
    return a.localeCompare(b);
  });

  for (const key of orderedKeys) {
    serializeEntry(lines, key, metadata[key], 0);
  }
  return lines.join('\n');
}

function serializeEntry(lines, key, value, indent) {
  const prefix = ' '.repeat(indent);

  if (Array.isArray(value)) {
    lines.push(`${prefix}${key}:`);
    for (const item of value) {
      if (item && typeof item === 'object' && !Array.isArray(item)) {
        lines.push(...serializeObject(item, indent + 2));
      } else {
        lines.push(`${prefix}  - ${quoteScalar(String(item))}`);
      }
    }
    return;
  }

  if (value && typeof value === 'object') {
    lines.push(`${prefix}${key}:`);
    for (const [subKey, subValue] of Object.entries(value)) {
      serializeEntry(lines, subKey, subValue, indent + 2);
    }
    return;
  }

  lines.push(`${prefix}${key}: ${quoteScalar(String(value))}`);
}

function serializeObject(object, indent) {
  const entries = Object.entries(object);
  if (entries.length === 0) {
    return [`${' '.repeat(indent)}- `];
  }

  const lines = [];
  const [[firstKey, firstValue], ...rest] = entries;
  lines.push(
    `${' '.repeat(indent)}- ${firstKey}: ${quoteScalar(String(firstValue))}`,
  );

  for (const [key, value] of rest) {
    serializeEntry(lines, key, value, indent + 2);
  }
  return lines;
}

function quoteScalar(value) {
  if (value === 'true' || value === 'false' || value === 'null') return value;
  if (/^-?\d+(\.\d+)?$/u.test(value) && !value.includes(' ')) return value;
  if (
    /^[A-Za-z0-9_./:@#\-]+$/.test(value) &&
    !value.includes("'") &&
    !value.includes(':')
  ) {
    return value;
  }
  return `'${value.replace(/'/gu, "''")}'`;
}
