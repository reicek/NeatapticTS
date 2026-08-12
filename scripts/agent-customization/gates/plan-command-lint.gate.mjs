#!/usr/bin/env node
/**
 * Tier-1 gate: plan-command-lint
 *
 * Validates shell commands referenced in plan Markdown files against actual
 * project CLI help output to catch flag drift (e.g., stale or misspelled
 * long flags) before plan updates are accepted.
 *
 * Gate contract: { pass: boolean, evidence: object, fixHint: string|null, owner: string }
 *
 * Usage:
 *   node scripts/agent-customization/gates/plan-command-lint.gate.mjs [--json]
 *   node scripts/agent-customization/gates/plan-command-lint.gate.mjs --plan=plans/<Plan>.plans.md [--json]
 *   node scripts/agent-customization/gates/plan-command-lint.gate.mjs --all [--json]
 */

import { readdir, readFile } from 'node:fs/promises';
import path from 'node:path';
import { exec } from 'node:child_process';
import { promisify } from 'node:util';
import { pathToFileURL } from 'node:url';
import { parseArgs, repoRoot } from '../customization-utils.mjs';

const execAsync = promisify(exec);

const OWNER = 'plan-command-lint.gate.mjs';
const DEFAULT_PLAN = 'plans/orchestration-fixes.plans.md';
const HELP_TIMEOUT_MS = 30_000;

const isMain = import.meta.url === pathToFileURL(process.argv[1] ?? '').href;

/**
 * Map of recognized CLI names to the help command used to fetch their
 * supported flags. The tsc entry uses `--help --all` so flags such as
 * `--skipLibCheck` are included in the output.
 */
const KNOWN_CLIS = new Map([
  ['jest', ['npx', 'jest', '--help']],
  ['tsc', ['npx', 'tsc', '--help', '--all']],
  ['eslint', ['npx', 'eslint', '--help']],
  ['prettier', ['npx', 'prettier', '--help']],
]);

const SHELL_BINARIES = new Set([
  'npx',
  'npm',
  'node',
  'jest',
  'tsc',
  'eslint',
  'prettier',
]);

/**
 * CLI entry point.
 *
 * @param {string[]} [argv] - Raw CLI arguments.
 * @returns {Promise<object>} Gate contract.
 */
export async function main(argv = process.argv.slice(2)) {
  const options = parseArgs(argv);

  const hasPlanFlag = argv.some(
    (arg) => arg === '--plan' || arg.startsWith('--plan='),
  );
  if (!options.all && !hasPlanFlag) {
    options.plan = DEFAULT_PLAN;
  }

  if (options.help) {
    console.log(`plan-command-lint gate

Usage:
  node scripts/agent-customization/gates/plan-command-lint.gate.mjs [--json]
  node scripts/agent-customization/gates/plan-command-lint.gate.mjs --plan=plans/<Plan>.plans.md [--json]
  node scripts/agent-customization/gates/plan-command-lint.gate.mjs --all [--json]

Options:
  --plan   Path to a repo-relative Markdown plan file (defaults to ${DEFAULT_PLAN}).
  --all    Scan every root-level .plans.md file instead of a single plan.
  --json   Write machine-readable JSON to stdout.
  --help   Show this help text.`);
    process.exit(0);
  }

  const planPath = options.all ? null : options.plan;
  const result = await runPlanCommandLintGate(planPath);

  if (options.json) {
    console.log(JSON.stringify(result, null, 2));
  } else {
    console.log(result.pass ? 'PASS' : 'FAIL', 'plan-command-lint gate');
    if (!result.pass && result.fixHint) {
      console.log('fixHint:', result.fixHint);
    }
    if (result.evidence.warnings && result.evidence.warnings.length > 0) {
      for (const warning of result.evidence.warnings) {
        console.log('WARNING:', warning);
      }
    }
  }

  process.exitCode = result.pass ? 0 : 1;
  return result;
}

/* istanbul ignore next */
if (isMain) {
  main();
}

// ---------------------------------------------------------------------------

/**
 * Runs the plan-command-lint gate.
 *
 * Reads the specified plan file (or every root-level .plans.md file when no
 * path is provided), extracts shell commands, and validates long flags against
 * the corresponding CLI's --help output.
 *
 * @param {string|null} [planPath] - Optional repo-relative path to a single plan file.
 *   When `null` or omitted, every root-level `*.plans.md` file is scanned.
 * @returns {Promise<{pass: boolean, evidence: object, fixHint: string|null, owner: string}>}
 */
export async function runPlanCommandLintGate(planPath = DEFAULT_PLAN) {
  const context = { helpCache: new Map(), warnings: [] };

  // Step 1: Resolve plan files.
  const planFiles = await resolvePlanFiles(planPath, context);
  if (planFiles.length === 0) {
    const note = planPath
      ? `Plan file not found: ${planPath}`
      : 'No root-level .plans.md files found.';
    return {
      pass: true,
      evidence: {
        scannedPlans: [],
        commandsChecked: 0,
        issues: [],
        warnings: context.warnings,
        note,
      },
      fixHint: planPath ? `Verify the plan path: ${planPath}` : null,
      owner: OWNER,
    };
  }

  // Step 2: Extract shell commands.
  const allCommands = [];
  for (const { filePath, text } of planFiles) {
    const commands = extractCommandsFromPlan(text, filePath);
    for (const command of commands) {
      allCommands.push(command);
    }
  }

  // Step 3: Validate flags.
  const issues = [];
  for (const command of allCommands) {
    const issue = await validateCommand(command, context);
    if (issue) {
      issues.push(issue);
    }
  }

  const pass = issues.length === 0;

  return {
    pass,
    evidence: {
      scannedPlans: planFiles.map((planFile) => planFile.filePath),
      commandsChecked: allCommands.length,
      commands: allCommands.map((command) => ({
        plan: command.plan,
        source: command.source,
        command: command.raw,
      })),
      issues,
      warnings: context.warnings,
    },
    fixHint: pass
      ? null
      : `Found ${issues.length} invalid flag(s) in plan validation commands. First: ${issues[0].plan} — ${issues[0].command} (invalid: ${issues[0].invalidFlags.join(', ')})`,
    owner: OWNER,
  };
}

// ---------------------------------------------------------------------------
// Plan discovery
// ---------------------------------------------------------------------------

/**
 * Resolves the set of plan files to lint.
 *
 * @param {string|null} planPath - Single repo-relative plan file, or `null` to scan all.
 * @param {object} context - Shared gate context; `context.warnings` is populated on errors.
 * @returns {Promise<Array<{filePath: string, text: string}>>}
 */
export async function resolvePlanFiles(planPath, context) {
  if (planPath) {
    try {
      const absolutePath = path.join(repoRoot, planPath);
      const text = await readFile(absolutePath, 'utf8');
      return [{ filePath: planPath, text }];
    } catch (error) {
      context.warnings.push(
        `Could not read plan file ${planPath}: ${error.message}`,
      );
      return [];
    }
  }

  try {
    const entries = await readdir(path.join(repoRoot, 'plans'), {
      withFileTypes: true,
    });
    const names = entries
      .filter((entry) => entry.isFile() && entry.name.endsWith('.plans.md'))
      .map((entry) => entry.name)
      .toSorted();
    const reads = names.map(async (name) => {
      const filePath = path.join('plans', name);
      const text = await readFile(path.join(repoRoot, filePath), 'utf8');
      return { filePath, text };
    });
    return Promise.all(reads);
  } catch (error) {
    context.warnings.push(`Could not list plans directory: ${error.message}`);
    return [];
  }
}

// ---------------------------------------------------------------------------
// Command extraction
// ---------------------------------------------------------------------------

/**
 * Extracts candidate shell commands from a plan Markdown body.
 *
 * Looks for YAML list items, inline YAML scalar values, and inline backtick
 * spans that start with a recognized shell binary.
 *
 * @param {string} text - Plan file text.
 * @param {string} planPath - Repo-relative path to the plan file.
 * @returns {Array<{raw: string, command: string, source: string, plan: string}>}
 */
export function extractCommandsFromPlan(text, planPath) {
  const commands = [];
  const seen = new Set();

  const addCommand = (raw, source) => {
    const command = stripInlineComment(raw);
    if (!looksLikeShellCommand(command)) {
      return;
    }
    const key = `${source}:${command}`;
    if (seen.has(key)) {
      return;
    }
    seen.add(key);
    commands.push({ raw, command, source, plan: planPath });
  };

  // YAML list items: "- 'command'" or `- command`.
  const listItemPattern = /^[ \t]*-[ \t]+(?:["']?)([^"'\r\n`]+)(?:["']?)$/gm;
  let match;
  while ((match = listItemPattern.exec(text)) !== null) {
    addCommand(match[1].trim(), 'yaml-list');
  }

  // Inline YAML values: `key: 'command'` or `key: "command"`.
  const inlineValuePattern =
    /^[ \t]*[a-zA-Z_][a-zA-Z0-9_-]*[ \t]*:[ \t]+(?:["']?)([^"'\r\n`]+)(?:["']?)$/gm;
  while ((match = inlineValuePattern.exec(text)) !== null) {
    addCommand(match[1].trim(), 'yaml-inline');
  }

  // Inline backtick commands.
  const backtickPattern = /`([^`\r\n]+)`/g;
  while ((match = backtickPattern.exec(text)) !== null) {
    addCommand(match[1].trim(), 'backtick');
  }

  return commands;
}

/**
 * Removes shell/YAML inline comments from a command string.
 *
 * Anything after an unquoted `#` preceded by whitespace is discarded.
 *
 * @param {string} value - Raw command string.
 * @returns {string}
 */
export function stripInlineComment(value) {
  return value.replace(/\s+#.*$/, '').trim();
}

/**
 * Determines whether a string looks like an executable shell command.
 *
 * @param {string} command - Command string to inspect.
 * @returns {boolean}
 */
export function looksLikeShellCommand(command) {
  if (typeof command !== 'string' || command.length === 0) {
    return false;
  }
  const firstWord = command.split(/\s+/)[0];
  return SHELL_BINARIES.has(firstWord);
}

// ---------------------------------------------------------------------------
// Flag validation
// ---------------------------------------------------------------------------

/**
 * Validates the long flags in a single extracted command.
 *
 * @param {object} commandInfo - Extracted command object.
 * @param {object} context - Shared gate context with `helpCache` and `warnings`.
 * @returns {object|null} Issue object if invalid flags were found, otherwise `null`.
 */
export async function validateCommand(commandInfo, context) {
  const words = commandInfo.command.split(/\s+/);
  /* istanbul ignore next -- defensive: a trimmed command always yields at least one word */
  if (words.length === 0) {
    return null;
  }

  let cli = words[0];
  let args = words.slice(1);

  // Unwrap npx / npm run ... style wrappers.
  if (cli === 'npx' || cli === 'npm') {
    const cliIndex = args.findIndex((arg) => !arg.startsWith('-'));
    if (cliIndex === -1) {
      return null;
    }
    cli = args[cliIndex];
    args = args.slice(cliIndex + 1);
  }

  if (!KNOWN_CLIS.has(cli)) {
    return null;
  }

  const longFlags = args
    .filter((arg) => arg.startsWith('--'))
    .map((arg) => (arg.includes('=') ? arg.slice(0, arg.indexOf('=')) : arg));

  if (longFlags.length === 0) {
    return null;
  }

  const helpText = await fetchHelp(cli, context);
  if (!helpText) {
    return null;
  }

  const invalidFlags = longFlags.filter(
    (flag) => !isFlagInHelp(flag, helpText),
  );
  if (invalidFlags.length === 0) {
    return null;
  }

  return {
    plan: commandInfo.plan,
    source: commandInfo.source,
    command: commandInfo.raw,
    cli,
    invalidFlags,
  };
}

/**
 * Checks whether a long flag appears as a standalone token in CLI help text.
 * Prevents substring false positives such as `--testPathPattern` matching
 * inside `--testPathPatterns`.
 *
 * @private
 * @param {string} flag - Long flag to look for.
 * @param {string} helpText - CLI help output.
 * @returns {boolean}
 *
 * @example
 * ```js
 * isFlagInHelp('--config', '  --config <path>  Path to config file.\n  --configs');
 * // => true
 * ```
 */
function isFlagInHelp(flag, helpText) {
  const escaped = flag.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const pattern = new RegExp(`(?<![a-zA-Z0-9-])${escaped}(?![a-zA-Z0-9-])`);
  return pattern.test(helpText);
}

/**
 * Fetches (and caches) the help text for a recognized CLI.
 *
 * On failure, records a warning in `context.warnings` and returns an empty
 * string so the gate does not crash.
 *
 * @param {string} cli - Recognized CLI name.
 * @param {object} context - Shared gate context with `helpCache` and `warnings`.
 * @returns {Promise<string>}
 */
export async function fetchHelp(cli, context) {
  if (context.helpCache.has(cli)) {
    return context.helpCache.get(cli);
  }

  const helpCommand = KNOWN_CLIS.get(cli);
  if (!helpCommand) {
    const message = `Unknown CLI "${cli}"; cannot validate flags.`;
    context.warnings.push(message);
    context.helpCache.set(cli, '');
    return '';
  }

  try {
    const { stdout } = await execAsync(helpCommand.join(' '), {
      cwd: repoRoot,
      timeout: HELP_TIMEOUT_MS,
    });
    context.helpCache.set(cli, stdout);
    return stdout;
  } catch (error) {
    const message = `Could not retrieve help for ${cli} (${helpCommand.join(' ')}): ${error.message}`;
    context.warnings.push(message);
    context.helpCache.set(cli, '');
    return '';
  }
}
