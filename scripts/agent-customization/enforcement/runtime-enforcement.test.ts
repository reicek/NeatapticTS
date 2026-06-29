import { rm } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..');
const RUNTIME_ENFORCEMENT_MODULE_URL = pathToFileURL(
  path.join(
    REPO_ROOT,
    'scripts',
    'agent-customization',
    'enforcement',
    'runtime-enforcement.mjs',
  ),
).href;
const TEST_SESSION_ID = 'runtime-enforcement-test-session';
const TEST_CONTEXT_PATH = path.join(
  REPO_ROOT,
  'rag-index',
  'data',
  'hook-context',
  `hook-context-${TEST_SESSION_ID}.json`,
);

describe('runtime-enforcement.mjs', () => {
  afterEach(async () => {
    await rm(TEST_CONTEXT_PATH, { force: true });
  });

  it('writes a prepared action carrier', async () => {
    const carrier = runRuntimeEnforcementEval([
      `const preparedCarrier = await runtimeEnforcement.prepareRuntimeContext(${JSON.stringify(
        {
          sessionId: TEST_SESSION_ID,
          flowId: '04.scoped-fix',
          currentAgent: '04-implementing',
          delegatorChain: ['01-planning', '04-implementing'],
          requiredSkills: ['plan-alignment'],
          requiredSpecialists: [],
          planPath: 'plans/mcp-active-binding.plans.md',
          activePhase: '1',
          activeStep: '1',
          allowedActionClass: 'write',
          expectedToolName: 'apply_patch',
        },
      )});`,
      `const currentCarrier = await runtimeEnforcement.readRuntimeContext(${JSON.stringify(TEST_SESSION_ID)});`,
      'console.log(JSON.stringify(currentCarrier));',
    ]);

    expect(carrier?.preparedAction?.flowId).toBe('04.scoped-fix');
  });

  it('rejects mismatched expected tool names', async () => {
    const validationResult = runRuntimeEnforcementEval([
      `const preparedCarrier = await runtimeEnforcement.prepareRuntimeContext(${JSON.stringify(
        {
          sessionId: TEST_SESSION_ID,
          flowId: '04.scoped-fix',
          currentAgent: '04-implementing',
          delegatorChain: ['01-planning', '04-implementing'],
          requiredSkills: ['plan-alignment'],
          requiredSpecialists: [],
          planPath: 'plans/mcp-active-binding.plans.md',
          activePhase: '1',
          activeStep: '1',
          allowedActionClass: 'write',
          expectedToolName: 'apply_patch',
        },
      )});`,
      `const validation = runtimeEnforcement.validatePreparedRuntimeContext({
        carrier: preparedCarrier,
        toolName: 'edit',
        planPath: 'plans/mcp-active-binding.plans.md',
      });`,
      'console.log(JSON.stringify(validation));',
    ]);

    expect(validationResult.ok).toBe(false);
  });

  it('describes a recovery hint when a strict carrier is missing', () => {
    const diagnosisResult = runRuntimeEnforcementEval([
      `const diagnosis = runtimeEnforcement.diagnosePreparedRuntimeContext(${JSON.stringify(
        {
          sessionId: TEST_SESSION_ID,
          toolName: 'powershell',
          planPath: 'plans/mcp-active-binding.plans.md',
        },
      )});`,
      'console.log(JSON.stringify(diagnosis));',
    ]);

    expect(diagnosisResult).toEqual({
      ok: false,
      reason:
        'Missing runtime enforcement context carrier for session runtime-enforcement-test-session and strict execute action.',
      recoveryHint:
        'Refresh the runtime proof with `node scripts/agent-customization/enforcement/runtime-enforcement-context.mjs --prepare --session-id=runtime-enforcement-test-session --plan=plans/mcp-active-binding.plans.md --tool-name=powershell --action-class=execute` before retrying the strict action.',
      actionClass: 'execute',
      preparedAction: null,
    });
  });

  it('clears the prepared action after cleanup', async () => {
    const clearedCarrier = runRuntimeEnforcementEval([
      `const preparedCarrier = await runtimeEnforcement.prepareRuntimeContext(${JSON.stringify(
        {
          sessionId: TEST_SESSION_ID,
          flowId: '04.scoped-fix',
          currentAgent: '04-implementing',
          delegatorChain: ['01-planning', '04-implementing'],
          requiredSkills: ['plan-alignment'],
          requiredSpecialists: [],
          planPath: 'plans/mcp-active-binding.plans.md',
          activePhase: '1',
          activeStep: '1',
          allowedActionClass: 'write',
          expectedToolName: 'apply_patch',
        },
      )});`,
      `await runtimeEnforcement.clearPreparedRuntimeContext(${JSON.stringify(TEST_SESSION_ID)}, preparedCarrier.preparedAction?.actionId ?? null);`,
      `const currentCarrier = await runtimeEnforcement.readRuntimeContext(${JSON.stringify(TEST_SESSION_ID)});`,
      'console.log(JSON.stringify(currentCarrier));',
    ]);

    expect(clearedCarrier?.preparedAction).toBeNull();
  });

  it('counts only the trailing gate failures after a pass reset', () => {
    const failureCount = runRuntimeEnforcementEval([
      `const failureCount = runtimeEnforcement.countTrailingGateFailures(${JSON.stringify(
        [
          {
            eventType: 'gate-exception',
            sessionId: TEST_SESSION_ID,
            gateId: 'first',
          },
          {
            eventType: 'runtime-action-prepass',
            sessionId: TEST_SESSION_ID,
            actionId: 'reset-action',
          },
          {
            eventType: 'gate-exception',
            sessionId: TEST_SESSION_ID,
            gateId: 'second',
          },
        ],
      )}, ${JSON.stringify(TEST_SESSION_ID)});`,
      'console.log(JSON.stringify(failureCount));',
    ]);

    expect(failureCount).toBe(1);
  });
});

function runRuntimeEnforcementEval(lines: string[]) {
  const inlineScript = [
    `const runtimeEnforcement = await import(${JSON.stringify(RUNTIME_ENFORCEMENT_MODULE_URL)});`,
    ...lines,
  ].join('\n');
  const evaluationResult = spawnSync(
    process.execPath,
    ['--input-type=module', '--eval', inlineScript],
    {
      cwd: REPO_ROOT,
      encoding: 'utf8',
      timeout: 120000,
    },
  );

  return JSON.parse(evaluationResult.stdout);
}
