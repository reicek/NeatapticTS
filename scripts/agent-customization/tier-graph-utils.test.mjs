/**
 * @module tier-graph-utils.test
 * @description Coverage tests for tier-graph-utils.mjs.
 */
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import {
  collectTierInventory,
  runValidateAgentGraph,
  resolveExpectedTier,
  resolveTierLabel,
  parseTier,
  isAllowedDelegation,
  TIER_LABELS,
  AGENT_DIRECTORY,
} from './tier-graph-utils.mjs';

function makeAgentFile(dir, name, content) {
  const filePath = path.join(dir, '.github', 'agents', `${name}.agent.md`);
  mkdirSync(path.dirname(filePath), { recursive: true });
  writeFileSync(filePath, content, 'utf8');
  return filePath;
}

function agentFrontmatter(opts = {}) {
  const lines = ['---'];
  if (opts.name !== undefined) lines.push(`name: ${opts.name}`);
  if (opts.tier !== undefined) lines.push(`tier: ${opts.tier}`);
  if (opts['user-invocable'] !== undefined)
    lines.push(`user-invocable: ${opts['user-invocable']}`);
  if (opts.agents !== undefined)
    lines.push(`agents: [${opts.agents.join(', ')}]`);
  if (opts.model !== undefined) lines.push(`model: ${opts.model}`);
  lines.push('---', '');
  return lines.join('\n');
}

describe('tier-graph-utils coverage', () => {
  let tempDir;

  beforeEach(() => {
    tempDir = mkdtempSync(path.join(os.tmpdir(), 'tier-graph-'));
  });

  afterEach(() => {
    rmSync(tempDir, { recursive: true, force: true });
  });

  describe('parseTier', () => {
    it('parses number values', () => {
      assert.equal(parseTier(1), 1);
      assert.equal(parseTier(4), 4);
    });

    it('parses numeric string values', () => {
      assert.equal(parseTier('2'), 2);
      assert.equal(parseTier('3'), 3);
    });

    it('returns null for non-numeric values', () => {
      assert.equal(parseTier('abc'), null);
      assert.equal(parseTier(undefined), null);
      assert.equal(parseTier(null), null);
      assert.equal(parseTier(''), null);
    });

    it('returns null for out-of-range values', () => {
      assert.equal(parseTier(0), null);
      assert.equal(parseTier(5), null);
      assert.equal(parseTier(-1), null);
      assert.equal(parseTier('0'), null);
      assert.equal(parseTier('5'), null);
    });
  });

  describe('isAllowedDelegation', () => {
    it('allows Tier 1 to delegate to Tiers 2, 3, 4', () => {
      assert.equal(isAllowedDelegation(1, 2), true);
      assert.equal(isAllowedDelegation(1, 3), true);
      assert.equal(isAllowedDelegation(1, 4), true);
    });

    it('forbids Tier 1 to delegate to Tier 1', () => {
      assert.equal(isAllowedDelegation(1, 1), false);
    });

    it('allows Tier 2 to delegate to Tiers 3, 4', () => {
      assert.equal(isAllowedDelegation(2, 3), true);
      assert.equal(isAllowedDelegation(2, 4), true);
    });

    it('forbids Tier 2 to delegate to Tiers 1, 2', () => {
      assert.equal(isAllowedDelegation(2, 1), false);
      assert.equal(isAllowedDelegation(2, 2), false);
    });

    it('allows Tier 3 to delegate to Tier 4 only', () => {
      assert.equal(isAllowedDelegation(3, 4), true);
      assert.equal(isAllowedDelegation(3, 3), false);
      assert.equal(isAllowedDelegation(3, 2), false);
      assert.equal(isAllowedDelegation(3, 1), false);
    });

    it('forbids Tier 4 from delegating', () => {
      assert.equal(isAllowedDelegation(4, 1), false);
      assert.equal(isAllowedDelegation(4, 2), false);
      assert.equal(isAllowedDelegation(4, 3), false);
      assert.equal(isAllowedDelegation(4, 4), false);
    });

    it('returns false for unknown tiers', () => {
      assert.equal(isAllowedDelegation(0, 1), false);
      assert.equal(isAllowedDelegation(5, 1), false);
      assert.equal(isAllowedDelegation(null, 1), false);
    });
  });

  describe('resolveExpectedTier', () => {
    it('returns correct tiers for known names', () => {
      assert.equal(resolveExpectedTier('01-planning'), 1);
      assert.equal(resolveExpectedTier('implementation-executor'), 2);
      assert.equal(resolveExpectedTier('learning-event-capturer'), 4);
      assert.equal(resolveExpectedTier('some-specialist'), 3);
    });
  });

  describe('resolveTierLabel', () => {
    it('returns labels for valid tiers', () => {
      assert.equal(resolveTierLabel(1), TIER_LABELS[1]);
      assert.equal(resolveTierLabel(2), TIER_LABELS[2]);
      assert.equal(resolveTierLabel(3), TIER_LABELS[3]);
      assert.equal(resolveTierLabel(4), TIER_LABELS[4]);
    });

    it('returns Unknown Tier for invalid tiers', () => {
      assert.equal(resolveTierLabel(0), 'Unknown Tier');
      assert.equal(resolveTierLabel(5), 'Unknown Tier');
      assert.equal(resolveTierLabel(null), 'Unknown Tier');
    });
  });

  describe('collectTierInventory', () => {
    it('detects unknown subagent references', async () => {
      makeAgentFile(
        tempDir,
        '01-planning',
        agentFrontmatter({
          name: '01-planning',
          tier: 1,
          'user-invocable': true,
          agents: ['nonexistent-agent'],
        }),
      );

      const inventory = await collectTierInventory({ workspaceRoot: tempDir });
      const unknownIssues = inventory.violations.filter((v) =>
        v.message.includes('Unknown subagent'),
      );
      assert.ok(unknownIssues.length > 0);
    });

    it('detects delegation tier violations', async () => {
      makeAgentFile(
        tempDir,
        '01-planning',
        agentFrontmatter({
          name: '01-planning',
          tier: 1,
          'user-invocable': true,
          agents: ['02-researching'],
        }),
      );
      makeAgentFile(
        tempDir,
        '02-researching',
        agentFrontmatter({
          name: '02-researching',
          tier: 1,
          'user-invocable': true,
        }),
      );

      const inventory = await collectTierInventory({ workspaceRoot: tempDir });
      const violationIssues = inventory.violations.filter((v) =>
        v.message.includes('Delegation tier violation'),
      );
      assert.ok(violationIssues.length > 0);
    });

    it('detects delegation cycles', async () => {
      makeAgentFile(
        tempDir,
        '01-planning',
        agentFrontmatter({
          name: '01-planning',
          tier: 1,
          'user-invocable': true,
          agents: ['implementation-executor'],
        }),
      );
      makeAgentFile(
        tempDir,
        'implementation-executor',
        agentFrontmatter({
          name: 'implementation-executor',
          tier: 2,
          'user-invocable': false,
          agents: ['01-planning'],
        }),
      );

      const inventory = await collectTierInventory({ workspaceRoot: tempDir });
      const cycleIssues = inventory.violations.filter((v) =>
        v.message.includes('Delegation cycle detected'),
      );
      assert.ok(cycleIssues.length > 0);
    });

    it('detects missing or invalid tier', async () => {
      makeAgentFile(
        tempDir,
        '01-planning',
        agentFrontmatter({
          name: '01-planning',
          tier: 'abc',
          'user-invocable': true,
        }),
      );

      const inventory = await collectTierInventory({ workspaceRoot: tempDir });
      const tierIssues = inventory.violations.filter((v) =>
        v.message.includes('Missing or invalid `tier`'),
      );
      assert.ok(tierIssues.length > 0);
    });

    it('detects tier mismatch and groups multiple issues by path', async () => {
      makeAgentFile(
        tempDir,
        '01-planning',
        agentFrontmatter({
          name: '01-planning',
          tier: 2,
          'user-invocable': false,
        }),
      );

      const inventory = await collectTierInventory({ workspaceRoot: tempDir });
      const mismatchIssues = inventory.violations.filter((v) =>
        v.message.includes('Expected tier'),
      );
      assert.ok(mismatchIssues.length > 0);
      // Multiple issues for the same path exercise groupIssuesByPath
      const agent = inventory.agents[0];
      assert.ok(agent.violations.length >= 2);
    });

    it('detects Tier 1 without user-invocable: true', async () => {
      makeAgentFile(
        tempDir,
        '01-planning',
        agentFrontmatter({
          name: '01-planning',
          tier: 1,
          'user-invocable': false,
        }),
      );

      const inventory = await collectTierInventory({ workspaceRoot: tempDir });
      const uiIssues = inventory.violations.filter((v) =>
        v.message.includes('must set `user-invocable: true`'),
      );
      assert.ok(uiIssues.length > 0);
    });

    it('detects non-Tier 1 with user-invocable: true', async () => {
      makeAgentFile(
        tempDir,
        'implementation-executor',
        agentFrontmatter({
          name: 'implementation-executor',
          tier: 2,
          'user-invocable': true,
        }),
      );

      const inventory = await collectTierInventory({ workspaceRoot: tempDir });
      const uiIssues = inventory.violations.filter((v) =>
        v.message.includes('Only Tier 1'),
      );
      assert.ok(uiIssues.length > 0);
    });

    it('detects Tier 4 with delegatesTo', async () => {
      makeAgentFile(
        tempDir,
        'learning-event-capturer',
        agentFrontmatter({
          name: 'learning-event-capturer',
          tier: 4,
          'user-invocable': false,
          agents: ['some-agent'],
        }),
      );

      const inventory = await collectTierInventory({ workspaceRoot: tempDir });
      const delegateIssues = inventory.violations.filter((v) =>
        v.message.includes('Tier 4 auxiliaries may not delegate'),
      );
      assert.ok(delegateIssues.length > 0);
    });

    it('handles missing agent directory gracefully', async () => {
      const inventory = await collectTierInventory({ workspaceRoot: tempDir });
      assert.equal(inventory.agents.length, 0);
      assert.equal(inventory.summary.total, 0);
    });

    it('uses default name from filename when name is missing', async () => {
      const filePath = path.join(
        tempDir,
        '.github',
        'agents',
        '01-planning.agent.md',
      );
      mkdirSync(path.dirname(filePath), { recursive: true });
      writeFileSync(filePath, '---\ntier: 1\n---\n', 'utf8');

      const inventory = await collectTierInventory({ workspaceRoot: tempDir });
      assert.equal(inventory.agents[0].name, '01-planning');
    });

    it('defaults userInvocable, delegatesTo, and model when missing', async () => {
      const filePath = path.join(
        tempDir,
        '.github',
        'agents',
        '01-planning.agent.md',
      );
      mkdirSync(path.dirname(filePath), { recursive: true });
      writeFileSync(filePath, '---\ntier: 1\n---\n', 'utf8');

      const inventory = await collectTierInventory({ workspaceRoot: tempDir });
      assert.equal(inventory.agents[0].user_invocable, true);
      assert.deepEqual(inventory.agents[0].delegates_to, []);
      assert.equal(inventory.agents[0].model, null);
    });

    it('produces a valid runValidateAgentGraph report', async () => {
      makeAgentFile(
        tempDir,
        '01-planning',
        agentFrontmatter({
          name: '01-planning',
          tier: 1,
          'user-invocable': true,
        }),
      );

      const report = await runValidateAgentGraph({ workspaceRoot: tempDir });
      assert.equal(typeof report.ok, 'boolean');
      assert.ok(Array.isArray(report.graph));
      assert.ok(report.inventory);
    });

    it('computes summary counts correctly', async () => {
      makeAgentFile(
        tempDir,
        '01-planning',
        agentFrontmatter({
          name: '01-planning',
          tier: 1,
          'user-invocable': true,
          agents: ['implementation-executor'],
        }),
      );
      makeAgentFile(
        tempDir,
        'implementation-executor',
        agentFrontmatter({
          name: 'implementation-executor',
          tier: 2,
          'user-invocable': false,
        }),
      );

      const inventory = await collectTierInventory({ workspaceRoot: tempDir });
      assert.equal(inventory.summary.total, 2);
      assert.equal(inventory.summary.by_tier[1], 1);
      assert.equal(inventory.summary.by_tier[2], 1);
      assert.equal(inventory.summary.user_invocable_total, 1);
      assert.equal(inventory.summary.delegation_edges, 1);
    });

    it('returns Unknown Tier label for agents with null tier in summary', async () => {
      makeAgentFile(
        tempDir,
        '01-planning',
        agentFrontmatter({
          name: '01-planning',
          tier: 'bad',
          'user-invocable': true,
        }),
      );

      const inventory = await collectTierInventory({ workspaceRoot: tempDir });
      assert.equal(inventory.agents[0].tier_label, 'Unknown Tier');
    });

    it('recursively discovers agent files in nested subdirectories', async () => {
      makeAgentFile(
        tempDir,
        '01-planning',
        agentFrontmatter({
          name: '01-planning',
          tier: 1,
          'user-invocable': true,
        }),
      );
      // Create a nested subdirectory with an agent file
      const nestedDir = path.join(
        tempDir,
        '.github',
        'agents',
        'subdir',
        'nested',
      );
      mkdirSync(nestedDir, { recursive: true });
      writeFileSync(
        path.join(nestedDir, 'nested-agent.agent.md'),
        agentFrontmatter({
          name: 'nested-agent',
          tier: 2,
          'user-invocable': false,
        }),
        'utf8',
      );

      const inventory = await collectTierInventory({ workspaceRoot: tempDir });
      assert.equal(inventory.summary.total, 2);
      const nested = inventory.agents.find(
        (a) => a.name === 'nested-agent',
      );
      assert.ok(nested);
      assert.equal(nested.tier, 2);
    });

    it('ignores non-agent.md files in the agents directory', async () => {
      makeAgentFile(
        tempDir,
        '01-planning',
        agentFrontmatter({
          name: '01-planning',
          tier: 1,
          'user-invocable': true,
        }),
      );
      // Create a non-.agent.md file that should be skipped
      const agentsDir = path.join(tempDir, '.github', 'agents');
      writeFileSync(
        path.join(agentsDir, 'README.md'),
        '# Agents',
        'utf8',
      );

      const inventory = await collectTierInventory({ workspaceRoot: tempDir });
      assert.equal(inventory.summary.total, 1);
    });

    it('uses default workspaceRoot when called without arguments', async () => {
      // Call without arguments to cover the = {} default parameter branch
      const inventory = await collectTierInventory();
      assert.ok(Array.isArray(inventory.agents));
    });

    it('uses default workspaceRoot in runValidateAgentGraph when called without arguments', async () => {
      // Call without arguments to cover the = {} default parameter branch
      const report = await runValidateAgentGraph();
      assert.equal(typeof report.ok, 'boolean');
    });
  });
});