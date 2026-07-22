import fs from 'node:fs';

const content = fs.readFileSync('scripts/agent-customization/mcp/neataptic-workflow-mcp.test.ts', 'utf8');
const lines = content.split('\n');
const itBlocks = [];
let currentIt = null;
let currentItLine = 0;
let currentExpects = [];

for (let i = 0; i < lines.length; i++) {
  const line = lines[i];
  const trimmed = line.trim();
  const itMatch = trimmed.match(/^it\(['"](.+?)['"]\s*,/);
  if (itMatch) {
    if (currentIt) {
      itBlocks.push({ name: currentIt, line: currentItLine, expects: currentExpects });
    }
    currentIt = itMatch[1];
    currentItLine = i + 1;
    currentExpects = [];
  }
  if (currentIt) {
    const expectMatches = [...line.matchAll(/expect\(/g)];
    for (const m of expectMatches) {
      currentExpects.push({ line: i + 1, col: m.index });
    }
  }
}
if (currentIt) {
  itBlocks.push({ name: currentIt, line: currentItLine, expects: currentExpects });
}

console.log('Total it blocks:', itBlocks.length);
console.log('Total expect calls:', itBlocks.reduce((a, b) => a + b.expects.length, 0));
const multi = itBlocks.filter((b) => b.expects.length > 1);
console.log('Multi-expect it blocks:', multi.length);
multi.forEach((b) => {
  console.log(`line ${b.line}: ${b.name} (${b.expects.length} expects)`);
  b.expects.forEach((e) => console.log(`  ${e.line}:${e.col}`));
});
