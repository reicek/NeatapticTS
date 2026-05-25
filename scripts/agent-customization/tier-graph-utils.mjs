/**
 * @module tier-graph-utils
 *
 * Tier-graph model and validation helpers for the NeatapticTS agent-customization system.
 *
 * Defines the five-tier delegation hierarchy (Tier 0 = default VS Code agent,
 * Tiers 1–4 = custom agents), enforces agent tier assignments and
 * `user-invocable` flag rules, validates delegation-direction constraints,
 * detects delegation cycles, and produces the structured inventory reports
 * consumed by `tier-inventory.mjs`, `validate-agent-graph.mjs`, and the
 * Tier Enforcement Gate.
 *
 * Allowed delegation directions:
 * - Tier 1 → Tier 2, 3, or 4
 * - Tier 2 → Tier 3 or 4
 * - Tier 3 → Tier 4 only
 * - Tier 4 → no delegation (auxiliaries are leaf nodes)
 */
import { readdir, readFile } from 'node:fs/promises';
import path from 'node:path';

import {
  issue,
  normalizePath,
  parseFrontmatter,
  summarizeIssues,
} from './customization-utils.mjs';

/** Repo-relative path to the folder containing all `.agent.md` definition files. */
export const AGENT_DIRECTORY = '.github/agents';

/** Human-readable display names for each delegation tier (1–4). */
export const TIER_LABELS = Object.freeze({
  1: 'Numbered SDLC Orchestrators',
  2: 'Named coordinators / sub-orchestrators',
  3: 'Hidden scouts and specialists',
  4: 'Auxiliaries and one-shot helpers',
});

/**
 * Canonical names of the eight numbered SDLC orchestrators (Tier 1).
 * These are the only agents permitted to set `user-invocable: true`.
 */
export const TIER_1_AGENT_NAMES = new Set([
  '00-helping',
  '01-planning',
  '02-researching',
  '03-red-testing',
  '04-implementing',
  '05-green-testing',
  '06-documenting',
  '07-logging',
]);

/**
 * Canonical names of named coordinators and sub-orchestrators (Tier 2).
 * Must set `user-invocable: false` and may only delegate to Tiers 3 or 4.
 */
export const TIER_2_AGENT_NAMES = new Set([
  'planning-context-coordinator',
  'planning-risk-coordinator',
  'planning-test-strategy-coordinator',
  'research-codebase-coordinator',
  'implementation-pattern-coordinator',
  'green-test-failure-triage-coordinator',
  'helping-gap-resolution-coordinator',
  'helping-agent-maintenance-coordinator',
  'solid-split',
  'flappy-architecture-polish',
]);

/**
 * Canonical names of auxiliary / one-shot helper agents (Tier 4).
 * Must set `user-invocable: false` and may not delegate to any other agent.
 */
export const TIER_4_AGENT_NAMES = new Set([
  'acceptance-criteria-writer',
  'docs-example-writer',
  'learning-event-capturer',
  'file-change-summarizer',
]);

/**
 * Collects the full tier inventory from `.agent.md` files and validates every
 * agent's tier assignment, `user-invocable` flag, and delegation edges.
 *
 * Detected violations include: missing/invalid `tier` field, tier-assignment
 * mismatch, wrong `user-invocable` value, unknown subagent names,
 * forbidden delegation directions, and delegation cycles.
 *
 * @param options - Optional configuration.
 * @param options.workspaceRoot - Absolute path to the repository root; defaults to `process.cwd()`.
 * @returns Structured report with `agents`, `violations`, and `summary` counts.
 */
export async function collectTierInventory({ workspaceRoot = process.cwd() } = {}) {
  const collectedAgents = await collectAgents({ workspaceRoot });
  const issues = [];
  const byName = new Map(collectedAgents.map((agent) => [agent.name, agent]));

  for (const agent of collectedAgents) {
    issues.push(...agent.parseIssues);
    issues.push(...validateAgentTierAssignment(agent));

    for (const childName of agent.delegatesTo) {
      const childAgent = byName.get(childName);
      if (!childAgent) {
        issues.push(issue('error', agent.file, `Unknown subagent '${childName}'.`));
        continue;
      }

      if (!isAllowedDelegation(agent.tier, childAgent.tier)) {
        issues.push(
          issue(
            'error',
            agent.file,
            `Delegation tier violation: Tier ${agent.tier} '${agent.name}' may not delegate to Tier ${childAgent.tier} '${childAgent.name}'.`,
          ),
        );
      }
    }
  }

  for (const cycle of findDelegationCycles(collectedAgents, byName)) {
    issues.push(issue('error', AGENT_DIRECTORY, `Delegation cycle detected: ${cycle.join(' -> ')}`));
  }

  const issuesByPath = groupIssuesByPath(issues);
  const agents = collectedAgents.map((agent) => ({
    file: agent.file,
    name: agent.name,
    tier: agent.tier,
    tier_label: resolveTierLabel(agent.tier),
    user_invocable: agent.userInvocable,
    delegates_to: agent.delegatesTo,
    model: agent.model,
    violations: issuesByPath.get(agent.file) ?? [],
  }));

  return {
    generated_at: new Date().toISOString(),
    agents,
    violations: issues,
    summary: {
      total: agents.length,
      by_tier: countAgentsByTier(agents),
      user_invocable_total: agents.filter((agent) => agent.user_invocable === true).length,
      delegation_edges: agents.reduce((total, agent) => total + agent.delegates_to.length, 0),
      violation_count: issues.length,
    },
  };
}

/**
 * Runs the full agent-graph validation and returns a structured report
 * suitable for JSON output or a human-readable summary.
 *
 * Delegates to {@link collectTierInventory} and reshapes the result into
 * the flat `graph` array format expected by `validate-agent-graph.mjs`.
 *
 * @param options - Optional configuration.
 * @param options.workspaceRoot - Absolute path to the repository root; defaults to `process.cwd()`.
 * @returns Validation report with `ok`, `graph`, `inventory`, and the raw violation list.
 */
export async function runValidateAgentGraph({ workspaceRoot = process.cwd() } = {}) {
  const inventory = await collectTierInventory({ workspaceRoot });

  return {
    ...summarizeIssues('agent graph', inventory.violations),
    generated_at: inventory.generated_at,
    graph: inventory.agents.map((agent) => ({
      name: agent.name,
      path: agent.file,
      tier: agent.tier,
      userInvocable: agent.user_invocable,
      agents: agent.delegates_to,
      violations: agent.violations,
    })),
    inventory: {
      summary: inventory.summary,
      violations: inventory.violations,
    },
  };
}

/**
 * Resolves the canonical tier for a given agent name based on the three
 * fixed allow-lists (Tier 1, 2, 4) and defaults to Tier 3 for all others.
 *
 * @param agentName - Agent filename stem (without the `.agent.md` extension).
 * @returns Expected tier number in the range 1–4.
 */
export function resolveExpectedTier(agentName) {
  if (TIER_1_AGENT_NAMES.has(agentName)) return 1;
  if (TIER_2_AGENT_NAMES.has(agentName)) return 2;
  if (TIER_4_AGENT_NAMES.has(agentName)) return 4;
  return 3;
}

/**
 * Returns the human-readable label for a tier number.
 *
 * @param tier - Tier number (1–4).
 * @returns Label string from {@link TIER_LABELS}, or `'Unknown Tier'` for out-of-range values.
 */
export function resolveTierLabel(tier) {
  return TIER_LABELS[tier] ?? 'Unknown Tier';
}

/**
 * Parses a raw frontmatter `tier` value into a validated integer (1–4).
 *
 * Accepts a number or a numeric string. Returns `null` when the value is
 * missing, non-numeric, or outside the 1–4 range.
 *
 * @param value - Raw YAML value from the `tier:` frontmatter field (may be `undefined`).
 * @returns Integer tier (1–4), or `null` when invalid.
 */
export function parseTier(value) {
  const parsedValue = typeof value === 'number' ? value : Number.parseInt(String(value ?? ''), 10);
  if (!Number.isInteger(parsedValue) || parsedValue < 1 || parsedValue > 4) {
    return null;
  }

  return parsedValue;
}

/**
 * Checks whether a Tier `parentTier` agent may delegate to a Tier `childTier` agent.
 *
 * Enforcement table:
 *
 * | Parent tier | Allowed child tiers |
 * |-------------|---------------------|
 * | 1           | 2, 3, 4             |
 * | 2           | 3, 4                |
 * | 3           | 4                   |
 * | 4           | none (leaf node)    |
 *
 * @param parentTier - Delegating agent's tier.
 * @param childTier - Target agent's tier.
 * @returns `true` when the delegation direction is permitted by the policy.
 */
export function isAllowedDelegation(parentTier, childTier) {
  switch (parentTier) {
    case 1:
      return childTier === 2 || childTier === 3 || childTier === 4;
    case 2:
      return childTier === 3 || childTier === 4;
    case 3:
      return childTier === 4;
    case 4:
      return false;
    default:
      return false;
  }
}

async function collectAgents({ workspaceRoot }) {
  const agentFiles = await listAgentFiles(workspaceRoot, path.join(workspaceRoot, AGENT_DIRECTORY));
  return Promise.all(agentFiles.map((absoluteFilePath) => readAgentDefinition({ absoluteFilePath, workspaceRoot })));
}

async function listAgentFiles(workspaceRoot, directoryPath) {
  let entries;
  try {
    entries = await readdir(directoryPath, { withFileTypes: true });
  } catch {
    return [];
  }

  const discoveredFiles = [];

  for (const entry of entries) {
    const absoluteEntryPath = path.join(directoryPath, entry.name);
    if (entry.isDirectory()) {
      discoveredFiles.push(...(await listAgentFiles(workspaceRoot, absoluteEntryPath)));
      continue;
    }

    if (entry.isFile() && entry.name.endsWith('.agent.md')) {
      discoveredFiles.push(absoluteEntryPath);
    }
  }

  return discoveredFiles.toSorted();
}

async function readAgentDefinition({ absoluteFilePath, workspaceRoot }) {
  const relativePath = normalizePath(path.relative(workspaceRoot, absoluteFilePath));
  const parsed = parseFrontmatter(await readFile(absoluteFilePath, 'utf8'), relativePath);
  const name = parsed.data.name ?? path.basename(relativePath, '.agent.md');

  return {
    file: relativePath,
    name,
    tier: parseTier(parsed.data.tier),
    rawTier: parsed.data.tier,
    userInvocable: parsed.data['user-invocable'] ?? true,
    delegatesTo: Array.isArray(parsed.data.agents) ? parsed.data.agents : [],
    model: parsed.data.model ?? null,
    parseIssues: parsed.issues,
  };
}

function validateAgentTierAssignment(agent) {
  const issues = [];
  const expectedTier = resolveExpectedTier(agent.name);

  if (agent.tier === null) {
    issues.push(issue('error', agent.file, 'Missing or invalid `tier` frontmatter; expected an integer from 1 to 4.'));
  } else if (agent.tier !== expectedTier) {
    issues.push(
      issue(
        'error',
        agent.file,
        `Expected tier ${expectedTier} (${resolveTierLabel(expectedTier)}), found ${String(agent.rawTier)}.`,
      ),
    );
  }

  if (expectedTier === 1 && agent.userInvocable !== true) {
    issues.push(issue('error', agent.file, 'Tier 1 numbered SDLC orchestrators must set `user-invocable: true`.'));
  }

  if (expectedTier !== 1 && agent.userInvocable !== false) {
    issues.push(issue('error', agent.file, 'Only Tier 1 numbered SDLC orchestrators may set `user-invocable: true`.'));
  }

  if (expectedTier === 4 && agent.delegatesTo.length > 0) {
    issues.push(issue('error', agent.file, 'Tier 4 auxiliaries may not delegate to other agents.'));
  }

  return issues;
}

function countAgentsByTier(agents) {
  return {
    1: agents.filter((agent) => agent.tier === 1).length,
    2: agents.filter((agent) => agent.tier === 2).length,
    3: agents.filter((agent) => agent.tier === 3).length,
    4: agents.filter((agent) => agent.tier === 4).length,
  };
}

function groupIssuesByPath(issues) {
  const issuesByPath = new Map();

  for (const currentIssue of issues) {
    if (!issuesByPath.has(currentIssue.path)) {
      issuesByPath.set(currentIssue.path, []);
    }

    issuesByPath.get(currentIssue.path).push(currentIssue);
  }

  return issuesByPath;
}

function findDelegationCycles(agents, byName) {
  const cycles = [];
  const visiting = new Set();
  const visited = new Set();

  for (const agent of agents) {
    visit(agent.name, []);
  }

  return cycles;

  function visit(agentName, stack) {
    if (visiting.has(agentName)) {
      const cycleStartIndex = stack.indexOf(agentName);
      cycles.push([...stack.slice(cycleStartIndex), agentName]);
      return;
    }

    if (visited.has(agentName)) return;

    const currentAgent = byName.get(agentName);
    if (!currentAgent) return;

    visiting.add(agentName);
    for (const childName of currentAgent.delegatesTo) {
      visit(childName, [...stack, agentName]);
    }
    visiting.delete(agentName);
    visited.add(agentName);
  }
}