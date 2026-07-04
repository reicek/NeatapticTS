import { readFileSync } from 'node:fs';

const skillPath = '.github/skills/capturing-learning-event/SKILL.md';
const agentPath = '.github/agents/07-logging.agent.md';

const skill = readFileSync(skillPath, 'utf8');
const agent = readFileSync(agentPath, 'utf8');

let failures = 0;

function fail(msg) {
  console.error(`FAIL: ${msg}`);
  failures += 1;
}

// AC-033: eventType enumeration includes spec-checklist and gate-run (plus existing types and constitution-update)
const schemaMatch = skill.match(/"eventType":\s*"([^"]+)"/);
if (!schemaMatch) {
  fail(
    'Could not find eventType enumeration in capturing-learning-event/SKILL.md schema.',
  );
} else {
  const types = schemaMatch[1].split('|');
  for (const t of ['spec-checklist', 'gate-run', 'constitution-update']) {
    if (!types.includes(t)) {
      fail(`eventType enumeration missing ${t}; got ${schemaMatch[1]}`);
    }
  }
}

// AC-034: example JSONL entries for spec-checklist and gate-run in code blocks
const codeBlocks = [
  ...skill.matchAll(/```(?:json|jsonl)\n([\s\S]*?)\n```/g),
].map((m) => m[1]);
function hasExample(type) {
  return codeBlocks.some((block) => block.includes(`"eventType": "${type}"`));
}
for (const t of ['spec-checklist', 'gate-run']) {
  if (!hasExample(t)) {
    fail(
      `No example JSONL entry for ${t} inside a code block in capturing-learning-event/SKILL.md.`,
    );
  }
}

// AC-035: 07-logging.agent.md Log Format section references recording constitution updates,
// spec-checklist runs, and gate-run events in learning-log.jsonl.
const logFormatHeading = '## Log Format';
const logFormatStart = agent.indexOf(logFormatHeading);
if (logFormatStart === -1) {
  fail('Could not find ## Log Format section in 07-logging.agent.md.');
} else {
  const nextHeading = agent.indexOf(
    '## ',
    logFormatStart + logFormatHeading.length,
  );
  const section =
    nextHeading === -1
      ? agent.slice(logFormatStart)
      : agent.slice(logFormatStart, nextHeading);
  for (const term of [
    'constitution',
    'spec-checklist',
    'gate-run',
    'learning-log.jsonl',
  ]) {
    if (!section.includes(term)) {
      fail(`Log Format section missing reference to "${term}".`);
    }
  }
}

if (failures) {
  console.error(`\n${failures} acceptance check(s) failed.`);
  process.exit(1);
}

console.log('PASS: AC-033, AC-034, AC-035.');
