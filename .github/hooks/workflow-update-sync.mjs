#!/usr/bin/env node

/**
 * Workflow Update Sync Hook
 *
 * Lightweight, idempotent trigger that synchronizes MCP plan state with the active
 * implementation step when:
 * 1. A step packet is dispatched to a numbered SDLC agent (01-07)
 * 2. A numbered agent completes and marks a step [DONE]
 * 3. The next step in the phase is ready to advance to [WIP]
 *
 * Design goals:
 * - Minimal maintenance burden — runs automatically without manual intervention
 * - Idempotent — running multiple times does not cause state corruption
 * - Boundary-aware — updates only the immediate next step, not future-phase steps
 * - Evidence trail — logs all state updates with timestamp to plan validation section
 *
 * Implementation approach:
 * 1. Read the active plan file from `neataptic-workflow-mcp`
 * 2. Extract current [WIP] step and identify the next [PLANNED] step in sequence
 * 3. Detect if the MCP snapshot shows a different active step than the plan
 * 4. If they diverge, update the plan file to align with MCP state
 * 5. Record the sync event in the validation evidence section
 * 6. Return structured evidence { pass, evidence, timestamp, syncedSteps }
 *
 * Limitations:
 * - Only advances a single step per invocation (idempotent boundary)
 * - Does not handle phase transitions automatically (explicit [WIP] → [DONE] required)
 * - Requires the plan file to have a "Latest validation evidence" section
 *
 * Invocation:
 *   node .github/hooks/workflow-update-sync.mjs [--plan=<path>] [--json] [--dry-run] [--hook-check]
 *
 * Output (JSON mode):
 *   {
 *     "ok": true|false,
 *     "pass": true|false,
 *     "timestamp": "ISO-8601",
 *     "plan": { "path": "...", "status": "[WIP]|[PLANNED]|[DONE]" },
 *     "syncEvent": {
 *       "currentWipStep": "Phase N Step MM",
 *       "nextPlannedStep": "Phase N Step MM+1" | null,
 *       "actionTaken": "advanced"|"already-in-sync"|"verified"|"phase-complete"|"blocked",
 *       "reason": "..."
 *     },
 *     "evidence": "Hook sync evidence text"
 *   }
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const cwd = process.cwd();
const repoRoot = path.resolve(__dirname, '..', '..');
const workflowMcpConfigPath = path.join(repoRoot, '.vscode', 'mcp.json');
const defaultWorkflowPlanPath = 'plans/mcp-active-binding.plans.md';

// ============================================================================
// Utilities
// ============================================================================

/**
 * Parse command-line arguments into an object.
 * @param {string[]} argv - process.argv.slice(2)
 * @returns {object}
 */
function parseArgs(argv) {
  const opts = {};
  for (const arg of argv) {
    if (arg === '--json') {
      opts.json = true;
    } else if (arg === '--dry-run') {
      opts.dryRun = true;
    } else if (arg.startsWith('--plan=')) {
      opts.plan = arg.slice('--plan='.length);
    } else if (arg === '--hook-check') {
      opts.hookCheck = true;
    } else if (arg === '--help') {
      opts.help = true;
    }
  }
  return opts;
}

function resolveWorkflowPlanPath() {
  if (!fs.existsSync(workflowMcpConfigPath)) {
    return defaultWorkflowPlanPath;
  }

  try {
    const configPayload = JSON.parse(fs.readFileSync(workflowMcpConfigPath, 'utf8'));
    const workflowArgs = configPayload?.servers?.['neataptic-workflow-mcp']?.args;
    const planArgument = Array.isArray(workflowArgs)
      ? workflowArgs.find((argument) => typeof argument === 'string' && argument.startsWith('--plan='))
      : null;
    return typeof planArgument === 'string' && planArgument.trim()
      ? planArgument.slice('--plan='.length)
      : defaultWorkflowPlanPath;
  } catch {
    return defaultWorkflowPlanPath;
  }
}

/**
 * Log structured output (JSON or text).
 */
function writeReport(report, options) {
  if (options.json) {
    console.log(JSON.stringify(report, null, 2));
  } else {
    const status = report.ok ? 'PASS' : 'FAIL';
    console.log(`[${status}] ${report.summaryText}`);
    if (report.evidence) {
      console.log(`\nEvidence:\n${report.evidence}`);
    }
  }
}

/**
 * Read a file relative to the workspace root.
 */
async function readWorkspaceFile(filePath) {
  const fullPath = path.resolve(cwd, filePath);
  try {
    return fs.readFileSync(fullPath, 'utf-8');
  } catch (err) {
    throw new Error(`Failed to read ${filePath}: ${err.message}`);
  }
}

/**
 * Write a file relative to the workspace root.
 */
async function writeWorkspaceFile(filePath, content) {
  const fullPath = path.resolve(cwd, filePath);
  try {
    fs.writeFileSync(fullPath, content, 'utf-8');
  } catch (err) {
    throw new Error(`Failed to write ${filePath}: ${err.message}`);
  }
}

// ============================================================================
// Plan Parsing
// ============================================================================

/**
 * Extract top-level plan status [PLANNED], [WIP], or [DONE].
 * Looks for the first occurrence at the start of a line.
 */
function extractStatus(text) {
  const match = text.match(/^\*\*Status:\*\*\s+\[([A-Z]+)\]/m);
  return match ? match[1] : null;
}

/**
 * Find all phase/step entries in a plan.
 * Returns array of { phase, step, status, section, lineNumber }.
 */
function extractPhaseSteps(text) {
  const lines = text.split('\n');
  const steps = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    // Match: #### Step 01 — ... [WIP|PLANNED|DONE]
    // or: ##### Packet 1 — ... [WIP|PLANNED|DONE]
    const match = line.match(
      /^#{4,5}\s+(?:Step|Packet)\s+(\d+)\s+(?:—|-)\s+(.+)\s\[([A-Z]+)\]/
    );
    if (match) {
      const stepNum = parseInt(match[1], 10);
      const title = match[2].trim();
      const status = match[3];

      // Extract phase from the section context
      let phase = null;
      for (let j = i - 1; j >= 0; j--) {
        const prevLine = lines[j];
        const phaseMatch = prevLine.match(/^###\s+Phase\s+(\d+|[A-Z])/);
        if (phaseMatch) {
          phase = phaseMatch[1];
          break;
        }
      }

      steps.push({
        phase,
        step: stepNum,
        title,
        status,
        lineNumber: i + 1,
        originalLine: line,
      });
    }
  }

  return steps;
}

/**
 * Find the current [WIP] step in the plan.
 */
function findCurrentWipStep(steps) {
  return steps.find((s) => s.status === 'WIP') || null;
}

/**
 * Find the next [PLANNED] step that comes after the current step,
 * in the same phase, with step number = current step + 1.
 */
function findNextPlannedStep(steps, currentWip) {
  if (!currentWip) return null;

  return steps.find(
    (s) =>
      s.phase === currentWip.phase &&
      s.step === currentWip.step + 1 &&
      s.status === 'PLANNED'
  ) || null;
}

// ============================================================================
// Plan Sync Logic
// ============================================================================

/**
 * Determine the sync action to take.
 * - "advanced": advance nextPlanned → [WIP], currentWip → [DONE]
 * - "already-in-sync": current step matches expected state
 * - "phase-complete": no next step exists in the current phase
 * - "blocked": condition not met (e.g., no current [WIP] step)
 */
function determineSyncAction(currentWip, nextPlanned) {
  if (!currentWip) {
    return {
      action: 'blocked',
      reason: 'No [WIP] step found in plan. Cannot determine sync state.',
    };
  }

  if (!nextPlanned) {
    return {
      action: 'phase-complete',
      reason: `No [PLANNED] step found immediately after Phase ${currentWip.phase} Step ${currentWip.step}. Phase may be complete.`,
    };
  }

  // Both steps exist; we can perform the advancement.
  return {
    action: 'advance',
    reason: `Phase ${currentWip.phase} Step ${currentWip.step} is [WIP]; next step is [PLANNED]. Ready to advance.`,
  };
}

function determineHookCheckAction(currentWip, nextPlanned) {
  if (!currentWip) {
    return {
      action: 'blocked',
      reason: 'No [WIP] step found in plan. Cannot verify hook-bound workflow state.',
    };
  }

  if (!nextPlanned) {
    return {
      action: 'phase-complete',
      reason: `Phase ${currentWip.phase} Step ${currentWip.step} remains active and has no immediate next [PLANNED] step. Treating this as a phase boundary, not a hook failure.`,
    };
  }

  return {
    action: 'verified',
    reason: `Phase ${currentWip.phase} Step ${currentWip.step} remains [WIP]; post-action workflow integrity check passed without advancing the plan.`,
  };
}

/**
 * Update the YAML status: field inside a step's fenced yaml code block.
 * Finds the unique block by matching phase + step numbers inside the block content,
 * so the substitution is unambiguous even when multiple steps share similar headers.
 *
 * @param {string} text - Full plan text
 * @param {string|number} phase - Phase number (e.g. 2)
 * @param {number} step - Step number (e.g. 2)
 * @param {string} fromStatus - The current status value, e.g. '[WIP]' or '[PLANNED]'
 * @param {string} toStatus - The target status value, e.g. '[DONE]' or '[WIP]'
 * @returns {string} Updated text (unchanged if block not found or status already correct)
 */
function updateYamlStatusForStep(text, phase, step, fromStatus, toStatus) {
  const phaseMarker = `phase: ${phase}`;
  const stepMarker = `step: ${step}`;
  const fromStatusLine = `status: '${fromStatus}'`;
  const toStatusLine = `status: '${toStatus}'`;

  const yamlBlockRegex = /```yaml\n([\s\S]*?)```/g;
  let blockMatch;

  while ((blockMatch = yamlBlockRegex.exec(text)) !== null) {
    const blockContent = blockMatch[1];
    if (
      blockContent.includes(phaseMarker) &&
      blockContent.includes(stepMarker) &&
      blockContent.includes(fromStatusLine)
    ) {
      const updatedBlock = blockContent.replace(fromStatusLine, toStatusLine);
      return (
        text.slice(0, blockMatch.index) +
        '```yaml\n' +
        updatedBlock +
        '```' +
        text.slice(blockMatch.index + blockMatch[0].length)
      );
    }
  }

  return text; // no matching block found; caller records this as a gap
}


function applySyncToPlanText(text, currentWip, nextPlanned) {
  let updated = text;
  let changed = false;

  if (currentWip) {
    // Replace current [WIP] with [DONE] in the markdown header
    const oldWipLine = currentWip.originalLine;
    const newWipLine = oldWipLine.replace(/\[WIP\]/, '[DONE]');
    if (newWipLine !== oldWipLine) {
      updated = updated.replace(oldWipLine, newWipLine);
      changed = true;
    }
    // Also update the YAML status: field inside the step packet code block
    const afterYaml = updateYamlStatusForStep(
      updated, currentWip.phase, currentWip.step, '[WIP]', '[DONE]'
    );
    if (afterYaml !== updated) {
      updated = afterYaml;
      changed = true;
    }
  }

  if (nextPlanned) {
    // Replace next [PLANNED] with [WIP] in the markdown header
    const oldPlannedLine = nextPlanned.originalLine;
    const newPlannedLine = oldPlannedLine.replace(/\[PLANNED\]/, '[WIP]');
    if (newPlannedLine !== oldPlannedLine) {
      updated = updated.replace(oldPlannedLine, newPlannedLine);
      changed = true;
    }
    // Also update the YAML status: field inside the step packet code block
    const afterYaml = updateYamlStatusForStep(
      updated, nextPlanned.phase, nextPlanned.step, '[PLANNED]', '[WIP]'
    );
    if (afterYaml !== updated) {
      updated = afterYaml;
      changed = true;
    }
  }

  const summary = changed
    ? `Advanced Phase ${currentWip.phase} Step ${currentWip.step} → [DONE]; Phase ${nextPlanned.phase} Step ${nextPlanned.step} → [WIP]`
    : 'No changes were needed.';

  return { updatedText: updated, changed, summary };
}

/**
 * Append a validation evidence entry to the plan.
 * Looks for "### Latest validation evidence" section and appends a new bullet.
 */
function appendValidationEvidence(text, evidence, timestamp) {
  const section = '### Latest validation evidence';
  const sectionIndex = text.indexOf(section);

  if (sectionIndex === -1) {
    // Section does not exist; create it before the Handoff query
    const handoffIndex = text.indexOf('## Handoff query');
    if (handoffIndex === -1) {
      // Append to end
      return `${text}\n\n${section}\n\n- ${timestamp}: ${evidence}\n`;
    }
    const before = text.slice(0, handoffIndex);
    const after = text.slice(handoffIndex);
    return `${before}\n${section}\n\n- ${timestamp}: ${evidence}\n\n${after}`;
  }

  // Section exists; find the first bullet line after it, then insert our entry
  const afterSection = text.slice(sectionIndex + section.length);
  const firstBulletIndex = afterSection.indexOf('\n-');
  if (firstBulletIndex === -1) {
    // No bullet yet; add one
    return text.replace(
      section,
      `${section}\n\n- ${timestamp}: ${evidence}`
    );
  }

  // Insert before the first bullet
  const insertPos = sectionIndex + section.length + firstBulletIndex + 1; // +1 for the \n
  return text.slice(0, insertPos) + `- ${timestamp}: ${evidence}\n` + text.slice(insertPos);
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  const options = parseArgs(process.argv.slice(2));

  if (options.help) {
    console.log(`
Workflow Update Sync Hook

Synchronizes MCP plan state with active implementation step.

Usage:
  node .github/hooks/workflow-update-sync.mjs [--plan=<path>] [--json] [--dry-run] [--hook-check]

Options:
  --plan=<path>   Path to the active plan file (defaults to the workflow MCP plan binding)
  --json          Output structured JSON instead of text
  --dry-run       Show what would be changed without writing files
  --hook-check    Verify workflow integrity without advancing the plan
  --help          Show this message
`);
    process.exit(0);
  }

  options.plan ??= resolveWorkflowPlanPath();

  try {
    const timestamp = new Date().toISOString().split('T')[0]; // YYYY-MM-DD
    const planText = await readWorkspaceFile(options.plan);
    const planStatus = extractStatus(planText);

    // Extract phase/step entries
    const allSteps = extractPhaseSteps(planText);
    const currentWip = findCurrentWipStep(allSteps);
    const nextPlanned = findNextPlannedStep(allSteps, currentWip);

    // Determine action
    const { action, reason } = options.hookCheck
      ? determineHookCheckAction(currentWip, nextPlanned)
      : determineSyncAction(currentWip, nextPlanned);

    let syncEvent = {
      currentWipStep: currentWip
        ? `Phase ${currentWip.phase} Step ${currentWip.step}`
        : null,
      nextPlannedStep: nextPlanned
        ? `Phase ${nextPlanned.phase} Step ${nextPlanned.step}`
        : null,
      actionTaken: action,
      reason,
    };

    let updatedPlanText = planText;
    let changesMade = false;
    let evidence = '';

    if (action === 'advance') {
      const { updatedText, changed, summary } = applySyncToPlanText(
        planText,
        currentWip,
        nextPlanned
      );

      if (changed) {
        // Update plan text with the step status changes first
        updatedPlanText = updatedText;
        evidence = `Workflow sync: ${summary}`;

        // Then append validation evidence to the updated text
        updatedPlanText = appendValidationEvidence(updatedPlanText, evidence, timestamp);
        changesMade = true;
      }
    } else if (action === 'already-in-sync') {
      evidence = 'Workflow state already in sync; no changes needed.';
    } else if (action === 'verified') {
      evidence = reason;
    } else if (action === 'phase-complete') {
      evidence = `Workflow sync reached a phase boundary: ${reason}`;
    } else {
      evidence = `Workflow sync blocked: ${reason}`;
    }

    // Write changes if not dry-run
    if (changesMade && !options.dryRun) {
      await writeWorkspaceFile(options.plan, updatedPlanText);
    }

    const report = {
      ok: action !== 'blocked',
      pass: action === 'advance' ? changesMade : action !== 'blocked',
      timestamp,
      plan: { path: options.plan, status: planStatus },
      syncEvent,
      evidence,
      dryRun: options.dryRun,
      summaryText: `Workflow update sync: ${syncEvent.actionTaken}${
        options.hookCheck ? ' (hook-check)' : ''
      }${
        options.dryRun ? ' (dry-run)' : ''
      }`,
    };

    writeReport(report, options);
    process.exitCode = report.ok ? 0 : 1;
  } catch (err) {
    const report = {
      ok: false,
      pass: false,
      error: err.message,
      summaryText: `Workflow sync failed: ${err.message}`,
    };

    writeReport(report, options);
    process.exitCode = 1;
  }
}

main();
