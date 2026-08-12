import fs from 'node:fs';
const raw = fs.readFileSync('coverage/coverage-final.json', 'utf8');
const data = JSON.parse(raw);
const key = Object.keys(data).find(k => k.endsWith('combat.ts'));
console.log(key);
const first = Object.entries(data[key].branchMap).slice(0, 3);
console.log(JSON.stringify(first, null, 2));
console.log('b sample line:', first[0]?.[1]);
