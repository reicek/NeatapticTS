const fs = require('fs');
const focus = new Set([
  'src/neat/neat.nge-lifecycle.ts',
  'src/neat/nge-juvenile/neat.nge-juvenile.apply.ts',
  'src/neat/nge-juvenile/neat.nge-juvenile.grow.ts',
  'src/neat/nge-juvenile/neat.nge-juvenile.focus.ts',
  'src/neat/nge-juvenile/neat.nge-juvenile.constants.ts',
  'src/neat/nge-evolution/neat.nge-evolution.reproduction.ts',
  'src/neat/nge-dna/neat.nge-dna.operator.ts',
  'src/neat/nge-dna/neat.nge-dna.bridge.ts'
]);
const text = fs.readFileSync('coverage/lcov.info','utf8');
const records = text.split('end_of_record').map(s=>s.trim()).filter(Boolean);
for(const rec of records){
  const map = {SF:'',LF:0,LH:0,BRF:0,BRH:0,FNF:0,FNH:0,DA:{},BR:{},FN:{}};
  for(const line of rec.split('\n')){
    const t=line.trim(); if(!t) continue;
    const k=t.slice(0,t.indexOf(':'));
    const v=t.slice(t.indexOf(':')+1);
    if(k==='SF') map.SF=v;
    else if(k==='LF') map.LF=+v;
    else if(k==='LH') map.LH=+v;
    else if(k==='BRF') map.BRF=+v;
    else if(k==='BRH') map.BRH=+v;
    else if(k==='FNF') map.FNF=+v;
    else if(k==='FNH') map.FNH=+v;
    else if(k==='DA'){ const [ln,hit]=v.split(','); map.DA[ln]=+hit; }
    else if(k==='BRDA'){ const [ln,bk,taken]=v.split(','); map.BR[ln]=map.BR[ln]||[]; map.BR[ln].push({block:bk,taken:isNaN(taken)?null:+taken}); }
    else if(k==='FN'){ const [hit,fn]=v.split(','); map.FN[fn]=+hit; }
  }
  if(!focus.has(map.SF)) continue;
  console.log('FILE: '+map.SF);
  console.log('  S '+map.LH+'/'+map.LF+'  B '+map.BRH+'/'+map.BRF+'  F '+map.FNH+'/'+map.FNF+'  L '+map.LH+'/'+map.LF);
  const uncoveredLines = Object.entries(map.DA).filter(([ln,hit])=>hit===0).map(([ln])=>+ln).sort((a,b)=>a-b);
  console.log('  uncovered lines: '+JSON.stringify(uncoveredLines));
  const uncoveredBranches=[];
  for(const [ln,brs] of Object.entries(map.BR)){
    for(const b of brs){ if(b.taken===0 || b.taken===null) uncoveredBranches.push({line:+ln,block:b.block,taken:b.taken}); }
  }
  uncoveredBranches.sort((a,b)=>a.line-b.line || a.block-b.block);
  console.log('  uncovered branches: '+JSON.stringify(uncoveredBranches));
  const uncoveredFns = Object.entries(map.FN).filter(([fn,hit])=>hit===0).map(([fn])=>fn);
  console.log('  uncovered functions: '+JSON.stringify(uncoveredFns));
}
