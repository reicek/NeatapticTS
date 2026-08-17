#!/usr/bin/env node
/**
 * Tier-1 gate: legacy-plan-format
 *
 * Scans all plan files in plans/ and plans/completed/ for any non-[DONE]
 * YAML block that still uses the legacy format (no expansion field, or
 * deprecated agent:/agent_file:). The gate fails until all active blocks are
 * migrated.
 *
 * Gate contract: { pass: boolean, evidence: object, fixHint: string, owner: string }
 *
 * Usage:
 *   node scripts/agent-customization/gates/legacy-plan-format.gate.mjs [--json]
 */

import { readdir, readFile, stat } from 'node:fs/promises';
import path from 'node:path';
import {
  parseArgs,
  parsePlanYamlBlock,
  repoRoot,
} from '../customization-utils.mjs';

const YAML_BLOCK_PATTERN = /```yaml\r?\n([\s\S]*?)```/g;
const HEADING_PATTERN =
  /^(?:### Phase .+? \[(?<phaseStatus>PLANNED|WIP|DONE)\]|#### Step .+? \[(?<stepStatus>PLANNED|WIP|DONE)\])/mu;
const MIGRATION_COMMAND =
  'node scripts/agent-customization/migrate-plan-format.mjs --plan=<plan-file>';

const options = parseArgs(process.argv.slice(2));
const result = await runLegacyFormatGate();

if (options.json) {
  console.log(JSON.stringify(result, null, 2));
} else {
  console.log(result.pass ? 'PASS' : 'FAIL', 'legacy-plan-format gate');
  if (!result.pass) console.log('fixHint:', result.fixHint);
}

process.exitCode = result.pass ? 0 : 1;

async function runLegacyFormatGate() {
  const planFiles = await collectPlanFiles();
  const legacyBlocks = [];
  let blocksChecked = 0;

  for (const planFile of planFiles) {
    const filePath = path.join(repoRoot, planFile);
    let text;
    try {
      text = await readFile(filePath, 'utf8');
    } catch (error) {
      continue;
    }

    YAML_BLOCK_PATTERN.lastIndex = 0;
    let match;
    while ((match = YAML_BLOCK_PATTERN.exec(text)) !== null) {
      const rawBlock = match[1];
      const headingStatus = findNearestHeadingStatus(text, match.index);
      const statusValue = extractStatusFromYaml(rawBlock);
      blocksChecked += 1;

      if (headingStatus === 'DONE') continue;
      if (statusValue === 'DONE') continue;
      // PlanUpdate blocks are audit logs, not step/phase metadata.
      if (/^\s*PlanUpdate:/mu.test(rawBlock)) continue;

      let metadata;
      try {
        metadata = parsePlanYamlBlock(rawBlock);
      } catch (error) {
        legacyBlocks.push({
          planFile,
          blockIndex: match.index,
          reason: `YAML parse error: ${String(error)}`,
        });
        continue;
      }

      if (metadata.phase === undefined) {
        continue;
      }

      if (isLegacyBlock(metadata)) {
        const reasons = [];
        if (!Object.hasOwn(metadata, 'expansion'))
          reasons.push('missing expansion field');
        if (Object.hasOwn(metadata, 'agent'))
          reasons.push('deprecated agent field');
        if (Object.hasOwn(metadata, 'agent_file'))
          reasons.push('deprecated agent_file field');

        legacyBlocks.push({
          planFile,
          blockIndex: match.index,
          reasons,
        });
      }
    }
  }

  const pass = legacyBlocks.length === 0;

  return {
    pass,
    evidence: {
      plansScanned: planFiles.length,
      blocksChecked,
      legacyBlocks,
    },
    fixHint: pass
      ? 'No legacy-format active blocks found.'
      : `Migrate legacy blocks with: ${MIGRATION_COMMAND}. ${legacyBlocks.length} block(s) need migration.`,
    owner: 'legacy-plan-format.gate.mjs',
  };
}

async function collectPlanFiles() {
  const planDirs = ['plans', 'plans/completed'];
  const files = [];

  for (const dir of planDirs) {
    const dirPath = path.join(repoRoot, dir);
    let entries;
    try {
      entries = await readdir(dirPath);
    } catch (error) {
      continue;
    }

    for (const entry of entries) {
      // Ignore hidden/temp files created by other tooling and tests.
      if (entry.startsWith('_') || entry.startsWith('.')) continue;
      if (!entry.endsWith('.plans.md')) continue;
      const fullPath = path.join(dirPath, entry);
      const entryStat = await stat(fullPath);
      if (!entryStat.isFile()) continue;
      files.push(path.join(dir, entry).replace(/\\/gu, '/'));
    }
  }

  return files;
}

function extractStatusFromYaml(rawBlock) {
  const match =
    /^status:\s*['"]?\[?(?<status>PLANNED|WIP|DONE)\]?['"]?\s*(?:#.*)?$/mu.exec(
      rawBlock,
    );
  return match?.groups?.status ?? null;
}

function findNearestHeadingStatus(text, blockIndex) {
  const before = text.slice(0, blockIndex);
  const lines = before.split(/\r?\n/);
  for (let index = lines.length - 1; index >= 0; index--) {
    const match = HEADING_PATTERN.exec(lines[index]);
    if (match) {
      const status = match.groups.phaseStatus ?? match.groups.stepStatus;
      return /* istanbul ignore next -- regex guarantees one group is defined */ status;
    }
  }
  return null;
}

function isLegacyBlock(metadata) {
  /* istanbul ignore next -- metadata is always an object (phase check at line 81 throws on null/undefined) */
  if (!metadata || typeof metadata !== 'object') return true;
  if (Object.hasOwn(metadata, 'agent') || Object.hasOwn(metadata, 'agent_file'))
    return true;
  if (!Object.hasOwn(metadata, 'expansion')) return true;
  return false;
}
