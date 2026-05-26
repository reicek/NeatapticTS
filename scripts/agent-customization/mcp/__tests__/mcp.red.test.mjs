import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { describe, it } from 'node:test';

import {
  createMcpServer,
  createSelfCheckReport,
  selfCheckError,
} from '../mcp-utils.mjs';
import { loadActivePlanContext } from '../mcp-plan-utils.mjs';

const MCP_CONFIG_PATH = path.resolve(import.meta.dirname, '../../../../.vscode/mcp.json');

describe('MCP hardening red contracts', () => {
  describe('RED contracts', () => {
    it('rejects archived completed plan arguments in .vscode/mcp.json', async () => {
      const mcpConfigText = await readFile(MCP_CONFIG_PATH, 'utf8');
      const mcpConfig = JSON.parse(mcpConfigText);
      const archivedPlanArgs = Object.values(mcpConfig.servers ?? {}).flatMap((serverConfig) =>
        Array.isArray(serverConfig?.args)
          ? serverConfig.args.filter(
              (argumentValue) =>
                typeof argumentValue === 'string' && argumentValue.startsWith('--plan=plans/completed/'),
            )
          : [],
      );

      assert.deepStrictEqual(archivedPlanArgs, []);
    });

    it('supports resources/list without rejecting', async () => {
      const server = createMcpServer({ serverName: 't', serverVersion: '0', tools: [] });
      const dispatchPromise = server.dispatch({ jsonrpc: '2.0', id: 1, method: 'resources/list' });

      await assert.doesNotReject(dispatchPromise);
    });

    it('supports prompts/list without rejecting', async () => {
      const server = createMcpServer({ serverName: 't', serverVersion: '0', tools: [] });
      const dispatchPromise = server.dispatch({ jsonrpc: '2.0', id: 1, method: 'prompts/list' });

      await assert.doesNotReject(dispatchPromise);
    });
  });

  describe('GREEN contracts', () => {
    it('returns the expected selfCheckError shape', () => {
      const errorIssue = selfCheckError('test-path', 'test-msg');

      assert.deepStrictEqual(errorIssue, {
        severity: 'error',
        path: 'test-path',
        message: 'test-msg',
      });
    });

    it('marks createSelfCheckReport as not ok when one error is present', () => {
      const errorIssue = selfCheckError('x', 'y');
      const report = createSelfCheckReport('test', [errorIssue]);

      assert.strictEqual(report.ok, false);
    });

    it('throws when loadActivePlanContext is given a completed plan archive', async () => {
      const completedPlanLoad = loadActivePlanContext('plans/completed/workspace-mcp-registration.plans.md');

      await assert.rejects(completedPlanLoad, /Expected exactly one \[WIP\] phase/);
    });

    it('loads active context when the validation heading includes validation gates', async () => {
      const activePlanContext = await loadActivePlanContext('plans/mcp-active-binding.plans.md');

      assert.deepStrictEqual(
        {
          phase: activePlanContext.activePhase.number,
          step: activePlanContext.activeStep.number,
          agent: activePlanContext.activeStep.metadata.agent,
        },
        { phase: 1, step: 1, agent: '00-helping' },
      );
    });
  });
});
