#!/usr/bin/env node
/**
 * @description Tier-1 gate: chrome-devtools-mcp-coverage.
 *
 * Validates that the `03-red-testing` and `05-green-testing` agents include the
 * `chrome-devtools-mcp` skill in their `skills` frontmatter array and reference
 * all three Chrome DevTools MCP specialists (`performance-trace-specialist`,
 * `browser-ui-specialist`, `browser-memory-specialist`) in their `agents`
 * frontmatter array.
 *
 * The gate accepts an optional `agentLoader` dependency-injection parameter so
 * unit tests can supply synthetic frontmatter without touching the filesystem.
 * When no loader is supplied, the gate reads the real `.github/agents/*.md`
 * files from the repository root.
 *
 * Gate contract: `{ pass: boolean, evidence: object, fixHint: string|null, owner: string }`
 *
 * Usage:
 *   node scripts/agent-customization/gates/chrome-devtools-mcp-coverage.gate.mjs [--json]
 *
 * @param {boolean} [--json] - Emit the standard gate JSON contract.
 * @returns {void} Exits 0 when the coverage prerequisites are green, 1 otherwise.
 */
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import {
  parseArgs,
  parseFrontmatter,
  readWorkspaceFile,
} from '../customization-utils.mjs';

const OWNER = 'chrome-devtools-mcp-workflow';

/**
 * Agent names that must carry the chrome-devtools-mcp skill and specialist
 * references. Kept as a named constant so the gate's coverage scope is legible
 * and adjustable in one place.
 */
const REQUIRED_AGENTS = ['03-red-testing', '05-green-testing'];

/**
 * Skill that must appear in each required agent's `skills` frontmatter array.
 */
const REQUIRED_SKILL = 'chrome-devtools-mcp';

/**
 * Specialists that must appear in each required agent's `agents` frontmatter
 * array. These are the three Chrome DevTools MCP specialists added in Phase 6.
 */
const REQUIRED_SPECIALISTS = [
  'performance-trace-specialist',
  'browser-ui-specialist',
  'browser-memory-specialist',
];

/**
 * Agents directory (repo-relative) where the `.agent.md` files live.
 */
const AGENTS_DIR = '.github/agents';

/**
 * Runs the chrome-devtools-mcp-coverage gate.
 *
 * @param {object} [options] - Optional configuration.
 * @param {function} [options.agentLoader] - Async function that receives an
 *   agent name and returns the raw `.agent.md` file contents. Defaults to
 *   reading the real file from `.github/agents/<name>.agent.md`.
 * @returns {Promise<object>} Standard gate contract:
 *   `{ pass, evidence, fixHint, owner }`.
 */
export async function runChromeDevToolsMcpCoverageGate(options = {}) {
  const agentLoader =
    options.agentLoader ?? createDefaultAgentLoader(AGENTS_DIR);

  const agentReports = [];
  for (const agentName of REQUIRED_AGENTS) {
    agentReports.push(await inspectAgent(agentName, agentLoader));
  }

  const missingSkillAgents = agentReports
    .filter((report) => !report.hasSkill)
    .map((report) => report.name);
  const missingSpecialistAgents = agentReports.flatMap((report) =>
    report.missingSpecialists.map((specialist) => ({
      agent: report.name,
      specialist,
    })),
  );

  const pass =
    missingSkillAgents.length === 0 && missingSpecialistAgents.length === 0;

  return {
    pass,
    evidence: {
      requiredAgents: REQUIRED_AGENTS,
      requiredSkill: REQUIRED_SKILL,
      requiredSpecialists: REQUIRED_SPECIALISTS,
      agentReports: agentReports.map((report) => ({
        name: report.name,
        hasSkill: report.hasSkill,
        missingSpecialists: report.missingSpecialists,
        skills: report.skills,
        agents: report.agents,
      })),
      missingSkillAgents,
      missingSpecialistAgents,
    },
    fixHint: pass
      ? null
      : buildFixHint(missingSkillAgents, missingSpecialistAgents),
    owner: OWNER,
  };
}

/**
 * Inspects a single agent's frontmatter for the required skill and specialists.
 *
 * @param {string} agentName - Agent name (e.g. `03-red-testing`).
 * @param {function} agentLoader - Async loader returning raw file contents.
 * @returns {Promise<object>} Report describing whether the skill and
 *   specialists are present.
 */
async function inspectAgent(agentName, agentLoader) {
  const rawContents = await agentLoader(agentName);
  const { data } = parseFrontmatter(
    rawContents,
    `${AGENTS_DIR}/${agentName}.agent.md`,
  );

  const skills = toArray(data.skills);
  const agents = toArray(data.agents);
  const hasSkill = skills.includes(REQUIRED_SKILL);
  const missingSpecialists = REQUIRED_SPECIALISTS.filter(
    (specialist) => !agents.includes(specialist),
  );

  return {
    name: agentName,
    hasSkill,
    missingSpecialists,
    skills,
    agents,
  };
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
 * Builds a human-readable fix hint naming the first missing skill or specialist.
 *
 * @param {string[]} missingSkillAgents - Agents missing the required skill.
 * @param {object[]} missingSpecialistAgents - Agents missing specialists.
 * @returns {string} Fix hint string.
 */
function buildFixHint(missingSkillAgents, missingSpecialistAgents) {
  if (missingSkillAgents.length > 0) {
    return (
      `Add '${REQUIRED_SKILL}' to the skills array of ` +
      `${missingSkillAgents.join(', ')} agent frontmatter.`
    );
  }
  const entries = missingSpecialistAgents.map(
    (entry) => `${entry.specialist} (${entry.agent})`,
  );
  return `Add missing specialist(s) to agents array: ${entries.join(', ')}.`;
}

/**
 * Creates the default agent loader that reads real files from the repo.
 *
 * @param {string} agentsDir - Repo-relative agents directory.
 * @returns {function} Async loader function.
 */
function createDefaultAgentLoader(agentsDir) {
  return async (agentName) =>
    readWorkspaceFile(`${agentsDir}/${agentName}.agent.md`);
}

const options = parseArgs(process.argv.slice(2));

async function main() {
  const report = await runChromeDevToolsMcpCoverageGate();

  if (options.json) {
    console.log(JSON.stringify(report, null, 2));
  } else {
    console.log(
      report.pass ? 'PASS' : 'FAIL',
      'chrome-devtools-mcp-coverage gate',
    );
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
