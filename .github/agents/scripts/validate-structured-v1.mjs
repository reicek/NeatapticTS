#!/usr/bin/env node
import fs from 'fs';
import process from 'process';

const args = process.argv.slice(2);
const file = args[0];

try {
  const content = file ? fs.readFileSync(file, 'utf8') : fs.readFileSync(0, 'utf8');

  // Determine agent tier from YAML frontmatter so the structured-v1 contract can
  // match the canonical quality contract (Tier 1 uses PHASE_COMPLETE / SUB_ORCHESTRATORS_USED;
  // lower tiers omit them).
  const frontmatterMatch = content.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  if (!frontmatterMatch) {
    console.error(JSON.stringify({ pass: false, errors: ['No YAML frontmatter block found'] }));
    process.exit(2);
  }
  const tierMatch = frontmatterMatch[1].match(/^tier:\s*(\d+)\s*$/m);
  if (!tierMatch) {
    console.error(JSON.stringify({ pass: false, errors: ['Missing or malformed tier in frontmatter'] }));
    process.exit(2);
  }
  const tier = Number(tierMatch[1]);

  const startToken = '```structured-v1';
  const start = content.indexOf(startToken);
  if (start === -1) {
    console.error(JSON.stringify({ pass: false, errors: ['No fenced structured-v1 block found'] }));
    process.exit(2);
  }
  const afterStart = content.indexOf('\n', start + startToken.length);
  const endFence = '\n```';
  const end = content.indexOf(endFence, afterStart + 1);
  if (end === -1) {
    console.error(JSON.stringify({ pass: false, errors: ['Malformed fenced block (missing closing ``` )'] }));
    process.exit(2);
  }
  const block = content.slice(afterStart + 1, end).trim();

  // Tier-specific required field sets must match the canonical contract in
  // .github/AGENT_QUALITY_CONTRACT.md.
  const requiredByTier = {
    1: [
      'OUTPUT_CONTRACT', 'TASK_STATUS', 'TIER', 'ROLE', 'TASK_RECEIVED',
      'FILES_READ', 'FILES_CHANGED', 'KEY_FINDINGS', 'ACTIONS_TAKEN', 'VALIDATION_EVIDENCE',
      'BLOCKERS', 'RISKS_OR_GAPS', 'LEARNING_EVENT_NEEDED', 'SUGGESTED_NEXT_AGENT',
      'PHASE_COMPLETE', 'SUB_ORCHESTRATORS_USED', 'SUMMARY'
    ],
    2: [
      'OUTPUT_CONTRACT', 'TASK_STATUS', 'TIER', 'ROLE', 'TASK_RECEIVED',
      'FILES_READ', 'FILES_CHANGED', 'KEY_FINDINGS', 'ACTIONS_TAKEN', 'VALIDATION_EVIDENCE',
      'SPECIALISTS_USED', 'HANDOFF', 'BLOCKERS', 'RISKS_OR_GAPS', 'LEARNING_EVENT_NEEDED',
      'SUGGESTED_NEXT_AGENT', 'SUMMARY'
    ],
    3: [
      'OUTPUT_CONTRACT', 'TASK_STATUS', 'TIER', 'ROLE', 'TASK_RECEIVED',
      'FILES_READ', 'FILES_CHANGED', 'KEY_FINDINGS', 'ACTIONS_TAKEN', 'VALIDATION_EVIDENCE',
      'HANDOFF', 'BLOCKERS', 'RISKS_OR_GAPS', 'LEARNING_EVENT_NEEDED', 'SUGGESTED_NEXT_AGENT',
      'SUMMARY'
    ],
    4: [
      'OUTPUT_CONTRACT', 'TASK_STATUS', 'TIER', 'ROLE', 'TASK_RECEIVED',
      'FILES_READ', 'FILES_CHANGED', 'KEY_FINDINGS', 'ACTIONS_TAKEN', 'BLOCKERS',
      'RISKS_OR_GAPS', 'LEARNING_EVENT_NEEDED', 'SUGGESTED_NEXT_AGENT', 'SUMMARY'
    ]
  };

  if (!requiredByTier[tier]) {
    console.error(JSON.stringify({ pass: false, errors: [`Unsupported tier value: ${tier}`] }));
    process.exit(2);
  }

  const requiredKeys = requiredByTier[tier];

  const errors = [];
  for (const key of requiredKeys) {
    const re = new RegExp('^' + key + ':', 'm');
    if (!re.test(block)) errors.push(`Missing required key: ${key}`);
  }

  // Quick check for OUTPUT_CONTRACT value
  const ocMatch = block.match(/^OUTPUT_CONTRACT:\s*(.+)$/m);
  if (!ocMatch) {
    errors.push('OUTPUT_CONTRACT missing or malformed');
  } else if (ocMatch[1].trim() !== 'structured-v1') {
    errors.push(`OUTPUT_CONTRACT must equal 'structured-v1' (found: ${ocMatch[1].trim()})`);
  }

  if (errors.length === 0) {
    console.log(JSON.stringify({ pass: true }));
    process.exit(0);
  } else {
    console.error(JSON.stringify({ pass: false, errors }));
    process.exit(3);
  }
} catch (err) {
  console.error(JSON.stringify({ pass: false, errors: [String(err)] }));
  process.exit(4);
}
