#!/usr/bin/env node
import {
  fileExists,
  issue,
  parseArgs,
  printUsage,
  readWorkspaceFile,
  summarizeIssues,
  writeReport,
} from './customization-utils.mjs';

const legacyRequiredMetadataKeys = [
  'phase',
  'agent',
  'agent_file',
  'status',
  'mode',
  'source_of_truth',
  'copy_paste',
  'requires_prior_phase',
  'next_phase',
];

const legacyRequiredSections = [
  '**User instruction:**',
  '**Objective:**',
  '**Context the agent must know:**',
  '**Execution steps:**',
  '**Stop conditions:**',
  '**Required validation:**',
  '**Plan update requirement:**',
];

const stepRequiredMetadataKeys = [
  'phase',
  'step',
  'agent',
  'agent_file',
  'status',
  'mode',
  'source_of_truth',
  'copy_paste',
  'next_step',
];

const stepRequiredSections = [
  '**User instruction:**',
  '**Step objective:**',
  '**Context the agent must know:**',
  '**Execution steps:**',
  '**Stop conditions:**',
  '**Required validation:**',
  '**Plan update requirement:**',
  '**Whole-step copy rule:**',
];

const forbiddenSections = [
  '**Copy-paste prompt:**',
];

const options = parseArgs(process.argv.slice(2));

if (options.help) {
  printUsage({
    title: 'Validate copy-pasteable plan phase step packets.',
    usage: 'node scripts/agent-customization/validate-plan-phase-packets.mjs [--json] [--plan=plans/Agentic_Workflow_Architecture.plans.md]',
    options: [['--plan=<path>', 'Plan file whose implementation phases should be validated.']],
  });
  process.exit(0);
}

const planPath = options.plan;
const planText = await readWorkspaceFile(planPath);
const issues = [];
const phases = [...extractPhaseBlocks(planText)].map((phaseBlock) => ({
  ...phaseBlock,
  stepBlocks: [...extractStepBlocks(phaseBlock.body)],
}));

if (phases.length === 0) {
  issues.push(issue('error', planPath, 'No implementation phase packets found.'));
}

for (const [phaseIndex, phaseBlock] of phases.entries()) {
  await validatePhase(phaseBlock, phaseIndex + 1);
}

const wipCount = phases.filter((phaseBlock) => phaseBlock.headingStatus === 'WIP').length;
if (wipCount !== 1) {
  issues.push(issue('warning', planPath, `Expected exactly one [WIP] phase, found ${wipCount}.`));
}

const report = {
  ...summarizeIssues('plan phase packets', issues),
  plan: planPath,
  phases: phases.map((phaseBlock) => ({
    phase: phaseBlock.headingPhase,
    title: phaseBlock.headingTitle,
    status: phaseBlock.headingStatus,
    schema: phaseBlock.stepBlocks.length > 0 ? 'step' : 'legacy',
    agent: phaseBlock.stepBlocks.find((stepBlock) => stepBlock.headingStatus === 'WIP')?.metadata.agent
      ?? phaseBlock.stepBlocks[0]?.metadata.agent
      ?? phaseBlock.metadata.agent
      ?? null,
  })),
};

writeReport(report, options);
process.exitCode = report.ok ? 0 : 1;

function* extractPhaseBlocks(text) {
  const implementationMatch = /## Implementation phases\s*(?<body>[\s\S]*?)(?=^## Validation gates)/m.exec(text);
  if (!implementationMatch?.groups) return;

  const implementationBody = implementationMatch.groups.body;
  const phasePattern = /^### Phase (?<phase>\d+) — (?<title>.+?) \[(?<status>PLANNED|WIP|DONE)]\s*$/gm;
  const matches = [...implementationBody.matchAll(phasePattern)];

  for (const [matchIndex, match] of matches.entries()) {
    if (!match.groups) continue;
    const bodyStart = (match.index ?? 0) + match[0].length;
    const nextMatch = matches.at(matchIndex + 1);
    const bodyEnd = nextMatch?.index ?? implementationBody.length;
    const phaseBody = implementationBody.slice(bodyStart, bodyEnd);
    const metadata = extractMetadata(phaseBody);
    yield {
      headingPhase: Number(match.groups.phase),
      headingTitle: match.groups.title,
      headingStatus: match.groups.status,
      body: phaseBody,
      metadata,
    };
  }
}

function* extractStepBlocks(phaseBody) {
  const stepPattern = /^#### Step (?<step>\d{2})\s*[:\-—]\s*(?<title>.+?) \[(?<status>PLANNED|WIP|DONE)]\s*$/gm;
  const matches = [...phaseBody.matchAll(stepPattern)];

  for (const [matchIndex, match] of matches.entries()) {
    if (!match.groups) continue;
    const bodyStart = (match.index ?? 0) + match[0].length;
    const nextMatch = matches.at(matchIndex + 1);
    const bodyEnd = nextMatch?.index ?? phaseBody.length;
    const stepBody = phaseBody.slice(bodyStart, bodyEnd);
    yield {
      headingStep: Number(match.groups.step),
      headingTitle: match.groups.title,
      headingStatus: match.groups.status,
      body: stepBody,
      metadata: extractMetadata(stepBody),
    };
  }
}

async function validatePhase(phaseBlock, expectedPhase) {
  const phasePath = `${planPath}#phase-${phaseBlock.headingPhase}`;

  if (phaseBlock.headingPhase !== expectedPhase) {
    issues.push(issue('error', phasePath, `Expected phase ${expectedPhase}, found phase ${phaseBlock.headingPhase}.`));
  }

  if (phaseBlock.stepBlocks.length > 0) {
    await validateStepPhase(phaseBlock, phasePath);
    return;
  }

  await validateLegacyPhase(phaseBlock, phasePath);
}

async function validateLegacyPhase(phaseBlock, phasePath) {
  const hasYamlMetadata = /^\s*```yaml\r?\n/.test(phaseBlock.body);

  if (!hasYamlMetadata) {
    pushForbiddenSectionIssues(phaseBlock.body, phasePath, 'Phase packet must not contain a separate Copy-paste prompt section; the whole phase is the prompt.');

    if (phaseBlock.headingStatus !== 'DONE') {
      issues.push(issue('error', phasePath, 'Active or planned phases must use numbered step packets that start with Step 01 planning.'));
      return;
    }

    if (phaseBlock.body.trim().length === 0) {
      issues.push(issue('error', phasePath, 'Compressed done phase must keep a concise coverage note.'));
    }
    return;
  }

  if (phaseBlock.headingStatus !== 'DONE') {
    issues.push(issue('error', phasePath, 'Active or planned phases must use numbered step packets that start with Step 01 planning.'));
  }

  for (const key of legacyRequiredMetadataKeys) {
    if (!Object.hasOwn(phaseBlock.metadata, key)) {
      issues.push(issue('error', phasePath, `Missing metadata key: ${key}.`));
    }
  }

  const metadataPhase = Number(phaseBlock.metadata.phase);
  if (metadataPhase !== phaseBlock.headingPhase) {
    issues.push(issue('error', phasePath, `Metadata phase ${phaseBlock.metadata.phase ?? 'missing'} does not match heading.`));
  }

  const metadataStatus = stripStatus(phaseBlock.metadata.status);
  if (metadataStatus !== phaseBlock.headingStatus) {
    issues.push(issue('error', phasePath, `Metadata status ${phaseBlock.metadata.status ?? 'missing'} does not match heading.`));
  }

  if (phaseBlock.metadata.mode !== 'fresh-session') {
    issues.push(issue('error', phasePath, 'Metadata mode must be fresh-session.'));
  }

  if (phaseBlock.metadata.source_of_truth !== planPath) {
    issues.push(issue('error', phasePath, `Metadata source_of_truth must be ${planPath}.`));
  }

  if (phaseBlock.metadata.copy_paste !== 'true') {
    issues.push(issue('error', phasePath, 'Metadata copy_paste must be true.'));
  }

  if (phaseBlock.metadata.agent_file && !(await fileExists(phaseBlock.metadata.agent_file))) {
    issues.push(issue('error', phasePath, `Metadata agent_file does not exist: ${phaseBlock.metadata.agent_file}.`));
  }

  for (const section of legacyRequiredSections) {
    if (!phaseBlock.body.includes(section)) {
      issues.push(issue('error', phasePath, `Missing required section: ${section}`));
    }
  }

  pushForbiddenSectionIssues(phaseBlock.body, phasePath, 'Phase packet must not contain a separate Copy-paste prompt section; the whole phase is the prompt.');

  if (
    !phaseBlock.body.includes(`select \`${phaseBlock.metadata.agent}\``) ||
    !includesWordsInOrder(phaseBlock.body, 'paste this full phase packet')
  ) {
    issues.push(issue('error', phasePath, 'User instruction must tell the user to select the phase agent and paste the full phase packet.'));
  }
}

async function validateStepPhase(phaseBlock, phasePath) {
  if (phaseBlock.stepBlocks[0]?.headingStep !== 1) {
    issues.push(issue('error', phasePath, 'Step-based phases must start with Step 01.'));
  }

  const wipSteps = phaseBlock.stepBlocks.filter((stepBlock) => stepBlock.headingStatus === 'WIP').length;
  if (phaseBlock.headingStatus === 'WIP' && wipSteps !== 1) {
    issues.push(issue('error', phasePath, `Expected exactly one [WIP] step in active phase ${phaseBlock.headingPhase}, found ${wipSteps}.`));
  }

  if (phaseBlock.headingStatus !== 'WIP' && wipSteps !== 0) {
    issues.push(issue('error', phasePath, `Only [WIP] phases may contain [WIP] steps; found ${wipSteps} in phase ${phaseBlock.headingPhase}.`));
  }

  for (const [stepIndex, stepBlock] of phaseBlock.stepBlocks.entries()) {
    const expectedStep = stepIndex + 1;
    const stepPath = `${phasePath}-step-${String(stepBlock.headingStep).padStart(2, '0')}`;
    const hasYamlMetadata = /^\s*```yaml\r?\n/.test(stepBlock.body);

    if (stepBlock.headingStep !== expectedStep) {
      issues.push(issue('error', stepPath, `Expected step ${String(expectedStep).padStart(2, '0')}, found step ${String(stepBlock.headingStep).padStart(2, '0')}.`));
    }

    if (!hasYamlMetadata) {
      pushForbiddenSectionIssues(stepBlock.body, stepPath, 'Step packet must not contain a separate Copy-paste prompt section; the whole step is the prompt.');

      if (stepBlock.headingStatus !== 'DONE') {
        issues.push(issue('error', stepPath, 'Active or planned steps must keep a full step packet with yaml metadata.'));
        continue;
      }

      if (stepBlock.body.trim().length === 0) {
        issues.push(issue('error', stepPath, 'Compressed done step must keep a concise coverage note.'));
      }
      continue;
    }

    for (const key of stepRequiredMetadataKeys) {
      if (!Object.hasOwn(stepBlock.metadata, key)) {
        issues.push(issue('error', stepPath, `Missing metadata key: ${key}.`));
      }
    }

    const metadataPhase = Number(stepBlock.metadata.phase);
    if (metadataPhase !== phaseBlock.headingPhase) {
      issues.push(issue('error', stepPath, `Metadata phase ${stepBlock.metadata.phase ?? 'missing'} does not match phase heading.`));
    }

    const metadataStep = Number(stepBlock.metadata.step);
    if (metadataStep !== stepBlock.headingStep) {
      issues.push(issue('error', stepPath, `Metadata step ${stepBlock.metadata.step ?? 'missing'} does not match step heading.`));
    }

    const metadataStatus = stripStatus(stepBlock.metadata.status);
    if (metadataStatus !== stepBlock.headingStatus) {
      issues.push(issue('error', stepPath, `Metadata status ${stepBlock.metadata.status ?? 'missing'} does not match step heading.`));
    }

    if (stepBlock.metadata.mode !== 'fresh-session') {
      issues.push(issue('error', stepPath, 'Metadata mode must be fresh-session.'));
    }

    if (stepBlock.metadata.source_of_truth !== planPath) {
      issues.push(issue('error', stepPath, `Metadata source_of_truth must be ${planPath}.`));
    }

    if (stepBlock.metadata.copy_paste !== 'true') {
      issues.push(issue('error', stepPath, 'Metadata copy_paste must be true.'));
    }

    if (stepBlock.metadata.agent_file && !(await fileExists(stepBlock.metadata.agent_file))) {
      issues.push(issue('error', stepPath, `Metadata agent_file does not exist: ${stepBlock.metadata.agent_file}.`));
    }

    const expectedAgentPrefix = `${String(stepBlock.headingStep).padStart(2, '0')} `;
    if (!stepBlock.metadata.agent?.startsWith(expectedAgentPrefix)) {
      issues.push(issue('error', stepPath, `Step ${String(stepBlock.headingStep).padStart(2, '0')} must use the matching numbered agent.`));
    }

    for (const section of stepRequiredSections) {
      if (!stepBlock.body.includes(section)) {
        issues.push(issue('error', stepPath, `Missing required section: ${section}`));
      }
    }

    pushForbiddenSectionIssues(stepBlock.body, stepPath, 'Step packet must not contain a separate Copy-paste prompt section; the whole step is the prompt.');

    if (
      !stepBlock.body.includes(`select \`${stepBlock.metadata.agent}\``) ||
        !includesWordsInOrder(stepBlock.body, 'paste this full step packet')
    ) {
      issues.push(issue('error', stepPath, 'User instruction must tell the user to select the step agent and paste the full step packet.'));
    }
  }
}

  function includesWordsInOrder(body, phrase) {
    const escapedWords = phrase.split(/\s+/).map((word) => word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
    return new RegExp(escapedWords.join('\\s+')).test(body);
  }

function pushForbiddenSectionIssues(body, sectionPath, message) {
  for (const section of forbiddenSections) {
    if (body.includes(section)) {
      issues.push(issue('error', sectionPath, message));
    }
  }
}

function extractMetadata(phaseBody) {
  const match = /^\s*```yaml\r?\n(?<yaml>[\s\S]*?)\r?\n```/.exec(phaseBody);
  if (!match?.groups) return {};

  const metadata = {};
  for (const line of match.groups.yaml.split(/\r?\n/)) {
    const keyValueMatch = /^(?<key>[a-z_]+):\s*(?<value>.*)$/.exec(line);
    if (!keyValueMatch?.groups) continue;
    metadata[keyValueMatch.groups.key] = normalizeScalar(keyValueMatch.groups.value);
  }
  return metadata;
}

function normalizeScalar(value) {
  return value.trim().replace(/^['"]|['"]$/g, '');
}

function stripStatus(value) {
  return value?.replace(/^\[/, '').replace(/]$/, '') ?? null;
}