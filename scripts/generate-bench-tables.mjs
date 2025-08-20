/**
 * generate-bench-tables.mjs (ESM version)
 * Emits markdown tables derived from test/benchmarks/benchmark.results.json.
 */
import fs from 'fs';
import path from 'path';

function cell(v, w){
  const s = String(v ?? '');
  return s.length >= w ? s.slice(0,w) : s + ' '.repeat(w - s.length);
}
function fmtNum(n, digits=4){
  if (typeof n !== 'number' || !isFinite(n)) return '';
  return n.toFixed(digits).replace(/0+$/,'').replace(/\.$/,'');
}
function loadArtifact(){
  const file = path.resolve('test/benchmarks/benchmark.results.json');
  if(!fs.existsSync(file)){
    console.error('[bench:tables] Artifact not found:', file);
    return null;
  }
  try { return JSON.parse(fs.readFileSync(file,'utf8')); }
  catch(e){ console.error('[bench:tables] Parse error', e); return null; }
}
function normalizeAggregated(artifact){
  const raw = artifact.aggregated;
  if(Array.isArray(raw)){
    const out = { src:{}, dist:{} };
    for(const entry of raw){
      const mode = entry.mode;
      const sizeKey = String(entry.size);
      out[mode] = out[mode] || {};
      out[mode][sizeKey] = out[mode][sizeKey] || {}; // scenario bucket
      out[mode][sizeKey].all = out[mode][sizeKey].all || {};
      const metrics = ['buildMs','fwdAvgMs','bytesPerConn','heapUsed','rss'];
      out[mode][sizeKey].all = out[mode][sizeKey].all || {};
      for(const m of metrics){
        const meanKey = m + 'Mean';
        if(meanKey in entry){
          out[mode][sizeKey].all[m] = { mean: entry[meanKey] };
        }
      }
    }
    return out;
  }
  return raw || { src:{}, dist:{} };
}
function buildVariantDeltaTable(artifact){
  const agg = normalizeAggregated(artifact);
  const sizes = new Set(Object.keys(agg.src || {}));
  const ordered = Array.from(sizes).sort((a,b)=>parseInt(a)-parseInt(b));
  const rows = [];
  rows.push('Size    Metric            Src Mean      Dist Mean     Δ Abs         Δ %        Flag          Result');
  rows.push('------  ----------------- ------------- ------------- ------------- ---------- ------------- ------------------------------------------------------------');
  const metrics = ['buildMs','fwdAvgMs','bytesPerConn'];
  for(const size of ordered){
    for(const metric of metrics){
      const srcAgg = agg.src?.[size]?.all?.[metric];
      const distAgg = agg.dist?.[size]?.all?.[metric];
      const sMean = srcAgg?.mean; const dMean = distAgg?.mean;
      let dAbs, dPct;
      if(typeof sMean==='number' && typeof dMean==='number'){
        dAbs = dMean - sMean; dPct = sMean===0?undefined:(dAbs/sMean)*100;
      }
      const threshold = metric==='bytesPerConn'?3:5;
      const flag = typeof dPct==='number' ? (Math.abs(dPct) > threshold ? '(REG)' : 'OK') : '';
      const note = metric==='bytesPerConn'? (flag==='(REG)'?'Dist bytes divergence':'Parity') : (flag==='(REG)'?`Dist ${dAbs>0?'slower':'faster'} ${metric}`:'Parity');
      rows.push(
        cell(size,6)+'  '+
        cell(metric+'Mean',17)+' '+
        cell(typeof sMean==='number'?fmtNum(sMean,4):'',13)+' '+
        cell(typeof dMean==='number'?fmtNum(dMean,4):'',13)+' '+
        cell(typeof dAbs==='number'?(dAbs>=0?'+':'')+fmtNum(dAbs,4):'',13)+' '+
        cell(typeof dPct==='number'?(dPct>=0?'+':'')+fmtNum(dPct,2):'',10)+' '+
        cell(flag,13)+' '+
        note
      );
    }
  }
  return rows.join('\n');
}
function buildHeapTable(artifact){
  const agg = normalizeAggregated(artifact);
  const sizes = new Set(Object.keys(agg.src || {}));
  const ordered = Array.from(sizes).sort((a,b)=>parseInt(a)-parseInt(b));
  const rows = [];
  rows.push('Size    Metric         Src Bytes    Dist Bytes    Src MB    Dist MB   Δ Bytes      Δ %     Note');
  rows.push('------  -------------- ------------ ------------ --------- --------- ----------- ------- -------------------------------------------------');
  for(const size of ordered){
    for(const metric of ['heapUsed','rss']){
      const s = agg.src?.[size]?.all?.[metric]?.mean;
      const d = agg.dist?.[size]?.all?.[metric]?.mean;
      if(typeof s !== 'number' || typeof d !== 'number') continue;
      const dBytes = d - s; const dPct = s===0?0:(dBytes/s)*100;
      rows.push(
        cell(size,6)+'  '+
        cell(metric+'Mean',14)+' '+
        cell(s,12)+' '+
        cell(d,12)+' '+
        cell((s/1048576).toFixed(1),9)+' '+
        cell((d/1048576).toFixed(1),9)+' '+
        cell((dBytes>=0?'+':'')+dBytes,11)+' '+
        cell((dPct>=0?'+':'')+dPct.toFixed(2),7)+' '+
        'Informational'
      );
    }
  }
  return rows.join('\n');
}
function main(){
  const artifact = loadArtifact();
  if(!artifact) process.exit(1);
  const out = [
    '```',
    buildVariantDeltaTable(artifact),
    '```',
    '',
    '```',
    buildHeapTable(artifact),
    '```'
  ].join('\n');
  console.log(out);
}
main();
