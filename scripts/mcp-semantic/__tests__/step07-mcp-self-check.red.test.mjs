/**
 * @module step07-mcp-self-check.red.test
 * @description Red tests for Phase 6 Step 07 — MCP server registration and self-check.
 *
 * These tests define the EXPECTED behavior AFTER implementation. They must FAIL
 * because `runSelfCheck()` in `scripts/mcp-semantic/repo-cortex-mcp.mjs`
 * (lines 1429-1460) currently ONLY invokes the `index_stats` tool. It does NOT
 * verify that all 18 registered tools are callable and does NOT produce a
 * per-tool `tools_checked` report or a `tool_checks_count` field.
 *
 * Acceptance criteria targeted:
 * - "Self-check: all tools callable"
 * - "Self-check passes for all tools"
 *
 * Expected report shape AFTER implementation adds:
 * - `tools_checked`: array of `{ name: string, ok: boolean }` (18 entries).
 * - `tool_checks_count`: numeric count of tools checked (should be 18).
 *
 * Pure .mjs test — runs via Jest ESM project `mcp-semantic-mjs`.
 */

import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..');
const MCP_SCRIPT = path.join('scripts', 'mcp-semantic', 'repo-cortex-mcp.mjs');

/**
 * Sorted list of the 18 expected tool names registered in createRepoCortexTools().
 */
const EXPECTED_TOOL_NAMES = [
  'ann_build_index',
  'expand_query',
  'freshness_check',
  'index_stats',
  'list_families',
  'load_chunk',
  'load_document',
  'load_parent_chunk',
  'multi_hop_search',
  'parallel_search',
  'scan_code_quality',
  'search_advanced',
  'search_context',
  'search_corpus',
  'submit_feedback',
  'traverse_graph',
  'turso_branch',
  'turso_pitr',
];

/**
 * Run the MCP server self-check via the CLI entrypoint and return the parsed
 * JSON report.
 *
 * Uses `execFileSync` with a generous timeout because the self-check loads the
 * semantic index from disk.
 *
 * @returns {Record<string, unknown>} Parsed self-check report.
 */
function runSelfCheckReport() {
  const stdout = execFileSync('node', [MCP_SCRIPT, '--self-check', '--json'], {
    cwd: REPO_ROOT,
    encoding: 'utf8',
    timeout: 120000,
  });
  return JSON.parse(stdout);
}

describe('Phase 6 Step 07 — MCP server registration and self-check', () => {
  describe('self-check covers all registered tools', () => {
    it('includes a tools_checked entry for every registered tool', () => {
      const report = runSelfCheckReport();
      const checked = report.tools_checked ?? [];
      const checkedNames = checked.map((entry) => entry.name).sort();
      expect(checkedNames).toEqual(EXPECTED_TOOL_NAMES);
    });

    it('marks all 18 tools as callable in the self-check report', () => {
      const report = runSelfCheckReport();
      const checked = report.tools_checked ?? [];
      const allOk =
        checked.length === EXPECTED_TOOL_NAMES.length &&
        checked.every((entry) => entry.ok === true);
      expect(allOk).toBe(true);
    });

    it('reports tool_checks_count equal to the number of registered tools', () => {
      const report = runSelfCheckReport();
      expect(report.tool_checks_count).toBe(18);
    });
  });
});
