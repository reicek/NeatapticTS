#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  clearPreparedRuntimeContext,
  isRuntimeContextPreparation,
  readRuntimeContext,
  recordRuntimeActionEvent,
  requiresRuntimeProof,
  resolveSessionId,
  validatePreparedRuntimeContext,
} from '../enforcement/runtime-enforcement.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..', '..', '..');
const workflowSyncHookPath = path.join(
  repoRoot,
  '.github',
  'hooks',
  'workflow-update-sync.mjs',
);
const gateExceptionRecorderPath = path.join(
  repoRoot,
  'scripts',
  'agent-customization',
  'gates',
  'record-gate-exception.mjs',
);
const writeToolPattern =
  /^(apply_patch|edit|create|create_file|vscode_renameSymbol|editFiles|replace_string_in_file|createFile|writeFile)$/i;
const substantiveToolPattern =
  /^(apply_patch|powershell|task|edit|create|create_file|vscode_renameSymbol|editFiles|replace_string_in_file|createFile|writeFile)$/i;
const refreshSteps = Object.freeze([
  {
    name: 'build-index',
    args: ['rag-index/build-index.mjs', '--json-health'],
  },
  {
    name: 'build-browser-snapshot',
    args: ['rag-index/build-browser-snapshot.mjs', '--json'],
  },
  {
    name: 'prewarm-dense',
    args: ['rag-index/prewarm-dense.mjs', '--json'],
  },
  {
    name: 'cortex-index-gate',
    args: ['scripts/agent-customization/gates/cortex-index.gate.mjs', '--json'],
  },
]);

main().catch((error) => {
  process.stderr.write(
    `[cortex-auto-refresh] runtime enforcement failed before hook exit: ${error.message}\n`,
  );
  process.exit(2);
});

async function main() {
  const hookInput = readHookInput();
  const candidateText = collectCandidateStrings(hookInput).join('\n');
  if (isRuntimeContextPreparation(candidateText)) {
    writeHookOutput({ continue: true });
    return;
  }

  const shouldRunWorkflowIntegrity = shouldRunWorkflowSync(hookInput);
  const shouldRefresh = shouldRefreshCortex(hookInput);
  if (!shouldRunWorkflowIntegrity && !shouldRefresh) {
    writeHookOutput({ continue: true });
    return;
  }

  const stepSummaries = [];
  /* istanbul ignore next -- defensive: both tool_name and toolName undefined means shouldRefresh=false, so line 78 is unreachable */
  const toolName = String(hookInput.tool_name ?? hookInput.toolName ?? '');
  const runtimeProofRequired = requiresRuntimeProof(toolName);
  const sessionId = resolveSessionId(hookInput);
  let runtimePreparedAction = null;

  if (runtimeProofRequired) {
    const runtimeContext = await readRuntimeContext(sessionId);
    const runtimeValidation = validatePreparedRuntimeContext({
      carrier: runtimeContext,
      sessionId,
      toolName,
      planPath: null,
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
          runtimePreparedAction?.currentAgent ?? 'refresh-cortex-after-write',
        delegatorChain: runtimePreparedAction?.delegatorChain ?? [],
        requiredSkills: runtimePreparedAction?.requiredSkills ?? [],
        requiredSpecialists: runtimePreparedAction?.requiredSpecialists ?? [],
        planPath: runtimePreparedAction?.planPath ?? null,
        activePhase: runtimePreparedAction?.activePhase ?? null,
        activeStep: runtimePreparedAction?.activeStep ?? null,
        reason: runtimeValidation.reason,
        recoveryHint: runtimeValidation.recoveryHint,
      }).catch(() => {
        // Keep mismatch logging best-effort on the posttool path.
      });
      recordGateException({
        gateId: 'runtime-enforcement-posttool',
        agent: 'refresh-cortex-after-write',
        hookInput,
        toolName,
        stepResult: {
          status: 2,
          stderr: [runtimeValidation.reason, runtimeValidation.recoveryHint]
            .filter(Boolean)
            .join('\n'),
          stdout: '',
        },
        extraEvidence: {
          actionClass: runtimeValidation.actionClass,
          flowId: runtimePreparedAction?.flowId ?? null,
          currentAgent: runtimePreparedAction?.currentAgent ?? null,
          actionId: runtimePreparedAction?.actionId ?? null,
          recoveryHint: runtimeValidation.recoveryHint,
        },
      });
      process.stderr.write(
        formatFailure('runtime-enforcement-context', {
          stderr: runtimeValidation.reason,
          stdout: '',
          status: 2,
        }),
      );
      process.exit(2);
    }
  }

  /* istanbul ignore next -- false branch unreachable: writeToolPattern ⊂ substantiveToolPattern, so shouldRefresh=true implies shouldRunWorkflowIntegrity=true */
  if (shouldRunWorkflowIntegrity) {
    const workflowSyncStep = runNodeStep([
      workflowSyncHookPath,
      '--json',
      '--hook-check',
    ]);
    if (workflowSyncStep.status !== 0) {
      recordGateException({
        gateId: 'workflow-update-sync-posttool',
        agent: 'refresh-cortex-after-write',
        hookInput,
        toolName,
        stepResult: workflowSyncStep,
      });
      process.stderr.write(
        formatFailure('workflow-update-sync', workflowSyncStep),
      );
      process.exit(2);
    }

    stepSummaries.push({
      name: 'workflow-update-sync',
      summary: summarizeStdout(workflowSyncStep.stdout),
    });
  }

  if (shouldRefresh) {
    for (const step of refreshSteps) {
      const stepResult = runNodeStep(step.args);
      if (stepResult.status !== 0) {
        recordGateException({
          gateId: step.name,
          agent: 'refresh-cortex-after-write',
          hookInput,
          toolName,
          stepResult,
        });
        process.stderr.write(formatFailure(step.name, stepResult));
        process.exit(2);
      }

      stepSummaries.push({
        name: step.name,
        summary: summarizeStdout(stepResult.stdout),
      });
    }
  }

  if (runtimePreparedAction) {
    await recordRuntimeActionEvent({
      eventType: 'runtime-action-postpass',
      status: 'pass',
      sessionId,
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
      reason: 'Runtime enforcement post-action validation passed.',
    }).catch(() => {
      // Keep pass logging best-effort so the write path is not more brittle than the current hooks.
    });
    await clearPreparedRuntimeContext(
      sessionId,
      runtimePreparedAction.actionId,
    );
    stepSummaries.push({
      name: 'runtime-enforcement-context',
      summary: 'pass',
    });
  }

  writeHookOutput({
    continue: true,
    hookSpecificOutput: {
      hookEventName: 'PostToolUse',
      additionalContext: formatSuccessContext(stepSummaries),
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

function shouldRefreshCortex(hookInput) {
  const toolName = String(hookInput.tool_name ?? hookInput.toolName ?? '');
  return writeToolPattern.test(toolName);
}

function shouldRunWorkflowSync(hookInput) {
  const toolName = String(hookInput.tool_name ?? hookInput.toolName ?? '');
  return substantiveToolPattern.test(toolName);
}

function runNodeStep(stepArgs) {
  return spawnSync(process.execPath, stepArgs, {
    cwd: repoRoot,
    encoding: 'utf8',
    timeout: 180000,
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
      `[cortex-auto-refresh] failed to record gate exception for ${gateId}: ${summarizeFailure(recordStep)}\n`,
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

  /* istanbul ignore next -- split always returns ≥1 element, so at(-1) never returns undefined */
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
  if (
    typeof parsedOutput.documents === 'number' &&
    typeof parsedOutput.chunks === 'number'
  ) {
    return `${parsedOutput.documents} docs / ${parsedOutput.chunks} chunks`;
  }
  if (Array.isArray(parsedOutput.steps)) {
    return parsedOutput.steps
      .map((step) => `${step.name}:${step.status}`)
      .join(', ');
  }

  return 'ok';
}

function formatSuccessContext(stepSummaries) {
  const compactSummary = stepSummaries
    .map((stepSummary) => `${stepSummary.name}=${stepSummary.summary}`)
    .join(' | ');
  return `Post-tool enforcement completed: ${compactSummary}`;
}

function formatFailure(stepName, stepResult) {
  const stderrText = String(stepResult.stderr ?? '').trim();
  const stdoutText = String(stepResult.stdout ?? '').trim();
  const details =
    stderrText ||
    stdoutText ||
    `${stepName} exited with status ${stepResult.status ?? 'unknown'}.`;
  return `[cortex-auto-refresh] ${stepName} failed: ${details}\n`;
}

function writeHookOutput(payload) {
  process.stdout.write(`${JSON.stringify(payload)}\n`);
}
