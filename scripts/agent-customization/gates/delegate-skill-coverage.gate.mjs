#!/usr/bin/env node
/**
 * @description Tier-1 gate: delegate-skill-coverage.
 *
 * Validates that every Tier 1 and Tier 2 agent includes the `execute` skill in
 * its `skills` frontmatter array. The `execute` skill is the durable contract
 * that allows numbered orchestrators to delegate domain work to specialists, so
 * its presence on all orchestration-tier agents is a structural invariant.
 *
 * The gate accepts optional dependency-injection parameters so unit tests can
 * supply synthetic agent inventories without touching the filesystem. When no
 * loader is supplied, the gate reads the real `.github/agents/*.md` files from
 * the repository root.
 *
 * Gate contract: `{ pass: boolean, evidence: object, fixHint: string|null, owner: string }`
 *
 * Usage:
 *   node scripts/agent-customization/gates/delegate-skill-coverage.gate.mjs [--json]
 *
 * @param {boolean} [--json] - Emit the standard gate JSON contract.
 * @returns {void} Exits 0 when the execute coverage prerequisites are green, 1 otherwise.
 */
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import {
  listMarkdownFiles,
  parseArgs,
  parseFrontmatter,
  readWorkspaceFile,
} from '../customization-utils.mjs';

const OWNER = 'delegate-skill-workflow';

/**
 * Skill that must appear in each Tier 1 / Tier 2 agent's `skills` array.
 */
const REQUIRED_SKILL = 'execute';

/**
 * Tier values that require the execute skill. Tier 0 (user) and Tier 3+
 * (specialists) are excluded because they do not orchestrate delegation.
 */
const REQUIRED_TIERS = new Set([1, 2]);

/**
 * Agents directory (repo-relative) where the `.agent.md` files live.
 */
const AGENTS_DIR = '.github/agents';

/**
 * Runs the delegate-skill-coverage gate.
 *
 * @param {object} [options] - Optional configuration.
 * @param {function} [options.inventoryLoader] - Async function that returns an
 *   array of `{ name, relativePath, contents }` agent descriptors. Defaults to
 *   discovering real `.agent.md` files under `.github/agents/`.
 * @returns {Promise<object>} Standard gate contract:
 *   `{ pass, evidence, fixHint, owner }`.
 */
export async function runDelegateSkillCoverageGate(options = {}) {
  const inventoryLoader =
    options.inventoryLoader ?? createDefaultInventoryLoader(AGENTS_DIR);

  const inventory = await inventoryLoader();
  const agentReports = [];
  for (const descriptor of inventory) {
    const report = inspectAgent(descriptor);
    if (report !== null) agentReports.push(report);
  }

  const missingDelegateAgents = agentReports
    .filter((report) => !report.hasDelegate)
    .map((report) => report.name);

  const pass = missingDelegateAgents.length === 0;

  return {
    pass,
    evidence: {
      requiredSkill: REQUIRED_SKILL,
      requiredTiers: [...REQUIRED_TIERS].toSorted(),
      checkedAgentCount: agentReports.length,
      agentReports: agentReports.map((report) => ({
        name: report.name,
        tier: report.tier,
        hasDelegate: report.hasDelegate,
        skills: report.skills,
      })),
      missingDelegateAgents,
    },
    fixHint: pass
      ? null
      : `Add '${REQUIRED_SKILL}' to the skills array of: ${missingDelegateAgents.join(', ')}.`,
    owner: OWNER,
  };
}

/**
 * Inspects a single agent descriptor, returning a report when the agent is
 * Tier 1 or Tier 2, or `null` when the agent is outside the required tiers.
 *
 * @param {object} descriptor - Agent descriptor with `name`, `relativePath`, and `contents`.
 * @returns {object|null} Report object or `null` when the tier is not required.
 */
function inspectAgent(descriptor) {
  const { data } = parseFrontmatter(
    descriptor.contents,
    descriptor.relativePath,
  );

  const tier = resolveTier(data.tier);
  if (!REQUIRED_TIERS.has(tier)) return null;

  const skills = toArray(data.skills);
  return {
    name: descriptor.name,
    tier,
    hasDelegate: skills.includes(REQUIRED_SKILL),
    skills,
  };
}

/**
 * Resolves a frontmatter tier value to a finite integer. Non-numeric or missing
 * values resolve to `NaN`, which is never in the required-tiers set.
 *
 * @param {*} value - Parsed frontmatter value for the `tier` key.
 * @returns {number} Tier as a finite integer, or `NaN`.
 */
function resolveTier(value) {
  if (typeof value === 'number') return value;
  if (typeof value === 'string') {
    const parsed = Number.parseInt(value, 10);
    return Number.isFinite(parsed) ? parsed : Number.NaN;
  }
  return Number.NaN;
}

/**
 * Coerces a frontmatter value into a string array. Handles the case where the
 * parser returns a non-array value (e.g. a bare scalar or `true`).
 *
 * @param {*} value - Parsed frontmatter value.
 * @returns {string[]} Array of string entries (empty when value is not an array).
 */
function toArray(value) {
  return Array.isArray(value) ? value : [];
}

/**
 * Creates the default inventory loader that discovers and reads real `.agent.md`
 * files from the repo.
 *
 * @param {string} agentsDir - Repo-relative agents directory.
 * @returns {function} Async loader returning an array of agent descriptors.
 */
function createDefaultInventoryLoader(agentsDir) {
  return async () => {
    const filePaths = await listMarkdownFiles(agentsDir, (relativePath) =>
      relativePath.endsWith('.agent.md'),
    );

    const descriptors = [];
    for (const relativePath of filePaths) {
      const contents = await readWorkspaceFile(relativePath);
      const name = extractAgentName(relativePath);
      descriptors.push({ name, relativePath, contents });
    }
    return descriptors;
  };
}

/**
 * Extracts the agent name (without `.agent.md` suffix) from a repo-relative path.
 *
 * @param {string} relativePath - Repo-relative path like `.github/agents/03-red-testing.agent.md`.
 * @returns {string} Agent name like `03-red-testing`.
 */
function extractAgentName(relativePath) {
  const basename = path.basename(relativePath);
  return basename.replace(/\.agent\.md$/, '');
}

const options = parseArgs(process.argv.slice(2));

async function main() {
  const report = await runDelegateSkillCoverageGate();

  if (options.json) {
    console.log(JSON.stringify(report, null, 2));
  } else {
    console.log(report.pass ? 'PASS' : 'FAIL', 'delegate-skill-coverage gate');
    if (!report.pass) console.log('fixHint:', report.fixHint);
  }

  process.exitCode = report.pass ? 0 : 1;
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href
) {
  await main();
}