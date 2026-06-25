import fs from 'fs';
const content = fs.readFileSync(process.argv[2], 'utf8');
const lines = content.split('\n');
const grades = [];
for (const line of lines) {
  const t = line.trim();
  if (t.startsWith('{') && t.includes('"agent"')) {
    try {
      grades.push(JSON.parse(t));
    } catch (e) {}
  }
}
let sql = 'DELETE FROM agent_grades;\n';
for (const g of grades) {
  const recs = (g.recommendations || [])
    .map((r) => String(r).replace(/'/g, "''"))
    .join('; ');
  const agentName = String(g.agent).replace(/'/g, "''");
  sql += `INSERT INTO agent_grades (agent_name, tier, orchestration_score, tools_score, role_score, grade_round, status, recommendations) VALUES ('${agentName}', ${g.tier}, ${g.orchestration}, ${g.tools_skills}, ${g.role_knowledge}, 1, 'graded', '${recs}');\n`;
}
fs.writeFileSync('scripts/agent-grades.sql', sql);
console.log('Wrote ' + grades.length + ' INSERT statements');
