import { rm } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import path from 'node:path';

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..');
const PRETOOL_HOOK_PATH = path.join(
  REPO_ROOT,
  'scripts',
  'agent-customization',
  'hooks',
  'pretool-workflow-cortex-preflight.mjs',
);
const RUNTIME_CONTEXT_CLI_PATH = path.join(
  REPO_ROOT,
  'scripts',
  'agent-customization',
  'enforcement',
  'runtime-enforcement-context.mjs',
);
const TEST_SESSION_ID = 'runtime-hook-test-session';
const TEST_CONTEXT_PATH = path.join(
  REPO_ROOT,
  'rag-index',
  'data',
  'hook-context',
  `hook-context-${TEST_SESSION_ID}.json`,
);
const EDIT_HOOK_INPUT = JSON.stringify({
  tool_name: 'edit',
  parameters: {
    path: 'src\\neataptic.ts',
  },
});

const isCI = Boolean(process.env.CI || process.env.GITHUB_ACTIONS);
const describeOrSkip = isCI ? describe.skip : describe;

describeOrSkip('runtime enforcement hook path', () => {
  afterEach(async () => {
    await rm(TEST_CONTEXT_PATH, { force: true });
  });

  beforeAll(() => {
    // The hook also invokes the cortex-first-search gate, which fails when the
    // semantic index is stale. Refresh the index once so these enforcement tests
    // are self-contained and not coupled to the environment's index state.
    const refreshResult = spawnSync(
      process.execPath,
      ['rag-index/session-start-index.mjs'],
      {
        cwd: REPO_ROOT,
        encoding: 'utf8',
        timeout: 120000,
      },
    );
    if (refreshResult.status !== 0) {
      throw new Error(
        `Session-start index refresh failed before runtime hook tests:\n${refreshResult.stderr || refreshResult.stdout}`,
      );
    }
  });

  it('blocks a strict write action when no prepared runtime proof exists', () => {
    const hookResult = spawnSync(process.execPath, [PRETOOL_HOOK_PATH], {
      cwd: REPO_ROOT,
      env: {
        ...process.env,
        COPILOT_CLI_SESSION_ID: TEST_SESSION_ID,
      },
      encoding: 'utf8',
      input: EDIT_HOOK_INPUT,
      timeout: 120000,
    });
    expect(
      hookResult.status === 2 &&
        hookResult.stderr.includes(
          'Refresh the runtime proof with `node scripts/agent-customization/enforcement/runtime-enforcement-context.mjs --prepare --session-id=runtime-hook-test-session --tool-name=edit --action-class=write` before retrying the strict action.',
        ),
    ).toBe(true);
  });

  it('passes a strict write action when a prepared runtime proof exists', async () => {
    spawnSync(
      process.execPath,
      [
        RUNTIME_CONTEXT_CLI_PATH,
        '--prepare',
        `--session-id=${TEST_SESSION_ID}`,
        '--flow-id=04.scoped-fix',
        '--agent=04-implementing',
        '--delegator-chain=01-planning,04-implementing',
        '--required-skills=plan-alignment',
        '--required-specialists=',
        '--plan=plans/mcp-active-binding.plans.md',
        '--phase=1',
        '--step=1',
        '--tool-name=edit',
        '--action-class=write',
      ],
      {
        cwd: REPO_ROOT,
        encoding: 'utf8',
        timeout: 120000,
      },
    );
    const hookResult = spawnSync(process.execPath, [PRETOOL_HOOK_PATH], {
      cwd: REPO_ROOT,
      env: {
        ...process.env,
        COPILOT_CLI_SESSION_ID: TEST_SESSION_ID,
      },
      encoding: 'utf8',
      input: EDIT_HOOK_INPUT,
      timeout: 120000,
    });

    expect(
      hookResult.status === 0 &&
        hookResult.stdout.includes('runtime-enforcement-context=pass'),
    ).toBe(true);
  });

  it('passes a strict write action when the session override plan matches the prepared proof', async () => {
    spawnSync(
      process.execPath,
      [
        RUNTIME_CONTEXT_CLI_PATH,
        '--prepare',
        `--session-id=${TEST_SESSION_ID}`,
        '--flow-id=04.scoped-fix',
        '--agent=04-implementing',
        '--delegator-chain=01-planning,04-implementing',
        '--required-skills=plan-alignment',
        '--required-specialists=',
        '--plan=plans/completed/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md',
        '--phase=3',
        '--step=3',
        '--tool-name=edit',
        '--action-class=write',
      ],
      {
        cwd: REPO_ROOT,
        encoding: 'utf8',
        timeout: 120000,
      },
    );
    const hookResult = spawnSync(process.execPath, [PRETOOL_HOOK_PATH], {
      cwd: REPO_ROOT,
      env: {
        ...process.env,
        COPILOT_CLI_SESSION_ID: TEST_SESSION_ID,
      },
      encoding: 'utf8',
      input: EDIT_HOOK_INPUT,
      timeout: 120000,
    });

    expect(
      hookResult.status === 0 &&
        hookResult.stdout.includes('runtime-enforcement-context=pass'),
    ).toBe(true);
  });
});
