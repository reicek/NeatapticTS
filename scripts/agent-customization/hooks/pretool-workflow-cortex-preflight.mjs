#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  isRuntimeContextPreparation,
  readRuntimeContext,
  recordRuntimeActionEvent,
  requiresRuntimeProof,
  resolveSessionId,
  validatePreparedRuntimeContext,
} from '../enforcement/runtime-enforcement.mjs';
import { resolveEffectivePlanPath } from '../mcp/mcp-plan-utils.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..', '..', '..');
const workflowMcpPath = path.join(
  repoRoot,
  'scripts',
  'agent-customization',
  'mcp',
  'neataptic-workflow-mcp.mjs',
);
const workflowMcpConfigPath = path.join(repoRoot, '.vscode', 'mcp.json');
const gateExceptionRecorderPath = path.join(
  repoRoot,
  'scripts',
  'agent-customization',
  'gates',
  'record-gate-exception.mjs',
);
const defaultWorkflowPlanPath = 'plans/mcp-active-binding.plans.md';

/**
 * Conservative trigger rules:
 * - substantive tool names only,
 * - plan-bound work if the tool or payload mentions plans/.plans.md/.logs.md,
 * - corpus-bound work if the tool or payload targets repo source/doc/search surfaces.
 */
const substantiveToolPattern =
  /^(apply_patch|powershell|task|edit|create|create_file|createFile|editFiles|replace_string_in_file|writeFile|vscode_renameSymbol|rg|glob|view|neataptic-cortex-mcp\.(search_corpus|load_document|freshness_check)|neataptic-workflow-mcp\.get_active_workflow_snapshot)$/i;
const alwaysPlanBoundToolPattern =
  /^(apply_patch|powershell|task|neataptic-workflow-mcp\.get_active_workflow_snapshot)$/i;
const alwaysCorpusBoundToolPattern =
  /^(apply_patch|powershell|task|edit|create|create_file|createFile|editFiles|replace_string_in_file|writeFile|vscode_renameSymbol|rg|glob|neataptic-cortex-mcp\.(search_corpus|load_document|freshness_check))$/i;
const planPathPattern = /(^|[\\/])plans([\\/]|$)|\.plans\.md\b|\.logs\.md\b/i;
const corpusPathPattern =
  /(^|[\\/])(src|examples|benchmarks|testing|scripts|\.github|plans)([\\/]|$)|\.(ts|tsx|js|mjs|md|json)\b/i;

main().catch((error) => {
  process.stderr.write(
    `[workflow-cortex-preflight] runtime enforcement failed before hook exit: ${error.message}\n`,
  );
  process.exit(2);
});

async function main() {
  const hookInput = readHookInput();
  const toolName = String(hookInput.tool_name ?? hookInput.toolName ?? '');
  const candidateText = collectCandidateStrings(hookInput).join('\n');
  const isPlanBound =
    alwaysPlanBoundToolPattern.test(toolName) ||
    planPathPattern.test(candidateText);
  const isCorpusBound =
    alwaysCorpusBoundToolPattern.test(toolName) ||
    corpusPathPattern.test(candidateText);

  if (
    !substantiveToolPattern.test(toolName) ||
    (!isPlanBound && !isCorpusBound)
  ) {
    writeHookOutput({ continue: true });
    return;
  }

  const runtimeProofRequired =
    requiresRuntimeProof(toolName) &&
    !isRuntimeContextPreparation(candidateText);
  const workflowPlanPath = isPlanBound
    ? await resolveEffectivePlanPath({}, resolveWorkflowPlanPath())
    : null;
  let runtimePreparedAction = null;

  if (runtimeProofRequired) {
    const sessionId = resolveSessionId(hookInput);
    const runtimeContext = await readRuntimeContext(sessionId);
    const runtimeValidation = validatePreparedRuntimeContext({
      carrier: runtimeContext,
      sessionId,
      toolName,
      planPath: workflowPlanPath,
    });
    runtimePreparedAction = runtimeValidation.preparedAction;
    if (!runtimeValidation.ok) {
      await recordRuntimeActionEvent({
        eventType: 'runtime-proof-mismatch',
        status: 'blocked',
        sessionId,
        actionId: runtimePreparedAction?.actionId ?? null,
        toolName,
        actionClass: runtimeValidation.actionClass,
        flowId: runtimePreparedAction?.flowId ?? null,
        currentAgent:
          runtimePreparedAction?.currentAgent ??
          'pretool-workflow-cortex-preflight',
        delegatorChain: runtimePreparedAction?.delegatorChain ?? [],
        requiredSkills: runtimePreparedAction?.requiredSkills ?? [],
        requiredSpecialists: runtimePreparedAction?.requiredSpecialists ?? [],
        planPath: runtimePreparedAction?.planPath ?? workflowPlanPath,
        activePhase: runtimePreparedAction?.activePhase ?? null,
        activeStep: runtimePreparedAction?.activeStep ?? null,
        reason: runtimeValidation.reason,
        recoveryHint: runtimeValidation.recoveryHint,
      }).catch(() => {
        // Keep mismatch logging best-effort so the hook still returns the real block reason.
      });
      recordGateException({
        gateId: 'runtime-enforcement-context',
        agent: 'pretool-workflow-cortex-preflight',
        hookInput,
        toolName,
        stepResult: {
          status: 2,
          stderr: runtimeValidation.reason,
          stdout: '',
        },
        extraEvidence: {
          actionClass: runtimeValidation.actionClass,
          flowId: runtimePreparedAction?.flowId ?? null,
          currentAgent: runtimePreparedAction?.currentAgent ?? null,
          planPath: runtimePreparedAction?.planPath ?? workflowPlanPath,
          actionId: runtimePreparedAction?.actionId ?? null,
          recoveryHint: runtimeValidation.recoveryHint,
        },
      });
      process.stderr.write(
        formatFailure('runtime-enforcement-context', toolName, {
          status: 2,
          stderr: [runtimeValidation.reason, runtimeValidation.recoveryHint]
            .filter(Boolean)
            .join('\n'),
          stdout: '',
        }),
      );
      process.exit(2);
    }
  }

  const stepSummaries = [];
  if (isPlanBound) {
    const workflowStep = runNodeStep([
      workflowMcpPath,
      `--plan=${workflowPlanPath}`,
      '--self-check',
      '--json',
    ]);
    if (workflowStep.status !== 0) {
      recordGateException({
        gateId: 'workflow-mcp-self-check',
        agent: 'pretool-workflow-cortex-preflight',
        hookInput,
        toolName,
        stepResult: workflowStep,
      });
      process.stderr.write(
        formatFailure('workflow-mcp-self-check', toolName, workflowStep),
      );
      process.exit(2);
    }

    stepSummaries.push({
      name: 'workflow-mcp-self-check',
      summary: summarizeStdout(workflowStep.stdout),
    });
  }

  if (isCorpusBound) {
    const cortexStep = runNodeStep([
      'scripts/agent-customization/gates/cortex-first-search.gate.mjs',
      '--json',
    ]);
    if (cortexStep.status !== 0) {
      recordGateException({
        gateId: 'cortex-first-search-gate',
        agent: 'pretool-workflow-cortex-preflight',
        hookInput,
        toolName,
        stepResult: cortexStep,
      });
      process.stderr.write(
        formatFailure('cortex-first-search-gate', toolName, cortexStep),
      );
      process.exit(2);
    }

    stepSummaries.push({
      name: 'cortex-first-search-gate',
      summary: summarizeStdout(cortexStep.stdout),
    });
  }

  if (runtimePreparedAction) {
    await recordRuntimeActionEvent({
      eventType: 'runtime-action-prepass',
      status: 'pass',
      sessionId: resolveSessionId(hookInput),
      actionId: runtimePreparedAction.actionId,
      toolName,
      actionClass: runtimePreparedAction.allowedActionClass,
      flowId: runtimePreparedAction.flowId,
      currentAgent: runtimePreparedAction.currentAgent,
      delegatorChain: runtimePreparedAction.delegatorChain,
      requiredSkills: runtimePreparedAction.requiredSkills,
      requiredSpecialists: runtimePreparedAction.requiredSpecialists,
      planPath: runtimePreparedAction.planPath,
      activePhase: runtimePreparedAction.activePhase,
      activeStep: runtimePreparedAction.activeStep,
      reason: 'Runtime enforcement proof validated before tool execution.',
    }).catch(() => {
      // Keep pass logging best-effort so hooks do not become brittle on log I/O.
    });
    stepSummaries.push({
      name: 'runtime-enforcement-context',
      summary: 'pass',
    });
  }

  const boundaryLabel = [
    isPlanBound ? 'plan-bound' : null,
    isCorpusBound ? 'corpus-bound' : null,
  ]
    .filter(Boolean)
    .join(', ');

  writeHookOutput({
    continue: true,
    hookSpecificOutput: {
      hookEventName: 'PreToolUse',
      additionalContext: formatSuccessContext(
        toolName,
        boundaryLabel,
        stepSummaries,
      ),
    },
  });
}

function readHookInput() {
  const rawInput = readFileSync(0, 'utf8').trim();
  if (!rawInput) {
    return {};
  }

  try {
    return JSON.parse(rawInput);
  } catch {
    return {};
  }
}

function collectCandidateStrings(value, collectedStrings = []) {
  if (typeof value === 'string') {
    collectedStrings.push(value);
    return collectedStrings;
  }

  if (Array.isArray(value)) {
    for (const arrayValue of value) {
      collectCandidateStrings(arrayValue, collectedStrings);
    }
    return collectedStrings;
  }

  if (value && typeof value === 'object') {
    for (const nestedValue of Object.values(value)) {
      collectCandidateStrings(nestedValue, collectedStrings);
    }
  }

  return collectedStrings;
}

function resolveWorkflowPlanPath() {
  if (!existsSync(workflowMcpConfigPath)) {
    return defaultWorkflowPlanPath;
  }

  try {
    const configPayload = JSON.parse(
      readFileSync(workflowMcpConfigPath, 'utf8'),
    );
    const workflowArgs =
      configPayload?.servers?.['neataptic-workflow-mcp']?.args;
    const planArgument = Array.isArray(workflowArgs)
      ? workflowArgs.find(
          (argument) =>
            typeof argument === 'string' && argument.startsWith('--plan='),
        )
      : null;

    return typeof planArgument === 'string' && planArgument.trim()
      ? planArgument.slice('--plan='.length)
      : defaultWorkflowPlanPath;
  } catch {
    return defaultWorkflowPlanPath;
  }
}

function runNodeStep(stepArgs) {
  return spawnSync(process.execPath, stepArgs, {
    cwd: repoRoot,
    encoding: 'utf8',
    timeout: 120_000,
  });
}

function recordGateException({
  gateId,
  agent,
  hookInput,
  toolName,
  stepResult,
  extraEvidence = {},
}) {
  const evidence = {
    toolName,
    details: summarizeFailure(stepResult),
    ...extraEvidence,
  };
  const recordStep = runNodeStep([
    gateExceptionRecorderPath,
    '--json',
    `--gate-id=${gateId}`,
    `--agent=${agent}`,
    `--session-id=${resolveSessionId(hookInput)}`,
    `--evidence=${JSON.stringify(evidence)}`,
  ]);
  if (recordStep.status !== 0) {
    process.stderr.write(
      `[workflow-cortex-preflight] failed to record gate exception for ${gateId}: ${summarizeFailure(recordStep)}\n`,
    );
  }
}

function summarizeFailure(stepResult) {
  const stderrText = String(stepResult.stderr ?? '').trim();
  const stdoutText = String(stepResult.stdout ?? '').trim();
  return (
    stderrText ||
    stdoutText ||
    `Exited with status ${stepResult.status ?? 'unknown'}.`
  );
}

function summarizeStdout(stdoutText) {
  const trimmedText = String(stdoutText ?? '').trim();
  if (!trimmedText) {
    return 'ok';
  }

  const parsedOutput = parseTrailingJsonPayload(trimmedText);
  if (parsedOutput !== null) {
    return summarizeParsedOutput(parsedOutput);
  }

  return trimmedText.split(/\r?\n/).at(-1) ?? 'ok';
}

function parseTrailingJsonPayload(stdoutText) {
  const firstJsonBraceIndex = stdoutText.indexOf('{');
  const lastJsonBraceIndex = stdoutText.lastIndexOf('}');
  if (firstJsonBraceIndex === -1 || lastJsonBraceIndex === -1) {
    return null;
  }

  const candidateJson = stdoutText.slice(
    firstJsonBraceIndex,
    lastJsonBraceIndex + 1,
  );
  try {
    return JSON.parse(candidateJson);
  } catch {
    return null;
  }
}

function summarizeParsedOutput(parsedOutput) {
  if (typeof parsedOutput.pass === 'boolean') {
    return parsedOutput.pass ? 'pass' : 'fail';
  }
  if (typeof parsedOutput.ok === 'boolean') {
    return parsedOutput.ok ? 'ok' : 'not-ok';
  }
  if (typeof parsedOutput.status === 'string') {
    return parsedOutput.status;
  }
  if (Array.isArray(parsedOutput.issues)) {
    return parsedOutput.issues.length === 0
      ? 'ok'
      : `issues:${parsedOutput.issues.length}`;
  }

  return 'ok';
}

function formatSuccessContext(toolName, boundaryLabel, stepSummaries) {
  const compactSummary = stepSummaries
    .map((stepSummary) => `${stepSummary.name}=${stepSummary.summary}`)
    .join(' | ');
  return `Pre-tool workflow/Cortex preflight passed for ${toolName} (${boundaryLabel}): ${compactSummary}`;
}

function formatFailure(stepName, toolName, stepResult) {
  const stderrText = String(stepResult.stderr ?? '').trim();
  const stdoutText = String(stepResult.stdout ?? '').trim();
  const details =
    stderrText ||
    stdoutText ||
    `${stepName} exited with status ${stepResult.status ?? 'unknown'}.`;
  return `[workflow-cortex-preflight] ${stepName} failed before ${toolName}: ${details}\n`;
}

function writeHookOutput(payload) {
  process.stdout.write(`${JSON.stringify(payload)}\n`);
}
