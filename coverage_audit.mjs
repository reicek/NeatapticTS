import fs from 'node:fs';

const files = [
  'examples\\neatenstein\\browser-entry\\host\\game\\constants.ts',
  'examples\\neatenstein\\browser-entry\\host\\game\\combat.ts',
  'examples\\neatenstein\\browser-entry\\host\\game\\tick.ts',
  'examples\\neatenstein\\browser-entry\\host\\input.ts',
  'examples\\neatenstein\\browser-entry\\host\\game\\controls.ts',
  'examples\\neatenstein\\browser-entry\\renderer\\map.ts',
  'examples\\neatenstein\\browser-entry\\renderer\\raycast.ts',
  'examples\\neatenstein\\browser-entry\\renderer\\framebuffer.ts',
  'examples\\neatenstein\\browser-entry\\renderer\\walls.ts',
  'examples\\neatenstein\\browser-entry\\renderer\\sprites.ts',
  'examples\\neatenstein\\browser-entry\\constants.ts',
];

const raw = fs.readFileSync('coverage/coverage-final.json', 'utf8');
const data = JSON.parse(raw);
const root = process.cwd();

for (const rel of files) {
  const key = `${root}\\${rel}`;
  const entry = data[key];
  if (!entry) {
    console.log(`\nMISSING: ${rel}`);
    continue;
  }
  const s = entry.statementMap;
  const uncoveredStmts = [];
  for (const [id, loc] of Object.entries(s)) {
    if (entry.s[id] === 0) {
      uncoveredStmts.push(loc.start.line);
    }
  }
  const uncoveredBranches = [];
  if (entry.branchMap) {
    for (const [id, b] of Object.entries(entry.branchMap)) {
      const taken = entry.b[id];
      for (let i = 0; i < taken.length; i += 1) {
        if (taken[i] === 0) {
          uncoveredBranches.push({ line: b.loc.start.line, locLine: b.locations?.[i]?.start?.line ?? b.loc.start.line, type: b.type });
        }
      }
    }
  }
  const fnMap = entry.fnMap;
  const uncoveredFns = [];
  for (const [id, f] of Object.entries(fnMap)) {
    if (entry.f[id] === 0) {
      uncoveredFns.push(f.decl.start.line);
    }
  }
  console.log(`\n${rel}`);
  console.log(`  statements: ${entry.statementMap ? Object.keys(entry.statementMap).length : 0}, uncovered stmts: ${[...new Set(uncoveredStmts)].sort((a,b)=>a-b).join(', ')}`);
  console.log(`  functions: ${Object.keys(fnMap).length}, uncovered fn lines: ${[...new Set(uncoveredFns)].sort((a,b)=>a-b).join(', ')}`);
  console.log(`  uncovered branches (line): ${[...new Set(uncoveredBranches.map(b=>`${b.line}:${b.type}`))].sort().join('; ')}`);
}
