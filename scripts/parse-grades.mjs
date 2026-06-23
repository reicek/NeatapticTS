import fs from 'fs';
const content = fs.readFileSync(process.argv[2], 'utf8');
const lines = content.split('\n');
const grades = [];
for (const line of lines) {
  const t = line.trim();
  if (t.startsWith('{') && t.includes('"agent"')) {
    try {
      const obj = JSON.parse(t);
      grades.push(obj);
    } catch(e) {}
  }
}
console.log('Total graded: ' + grades.length);
grades.forEach(g => console.log(`${g.agent} T${g.tier}: orch=${g.orchestration} tools=${g.tools_skills} role=${g.role_knowledge}`));
const sum = {orch:0, tools:0, role:0};
grades.forEach(g => { sum.orch += g.orchestration; sum.tools += g.tools_skills; sum.role += g.role_knowledge; });
console.log(`\nAverages: orch=${(sum.orch/grades.length).toFixed(1)} tools=${(sum.tools/grades.length).toFixed(1)} role=${(sum.role/grades.length).toFixed(1)}`);