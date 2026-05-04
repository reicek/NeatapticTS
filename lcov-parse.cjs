const fs = require('fs');
const lines = fs.readFileSync('coverage/lcov.info','utf8').split('\n');
let sf='',lf=0,lh=0;
const results=[];
for(const line of lines){
  const l=line.trim();
  if(l.startsWith('SF:')) sf=l.slice(3);
  else if(l.startsWith('LF:')) lf=parseInt(l.slice(3));
  else if(l.startsWith('LH:')) lh=parseInt(l.slice(3));
  else if(l==='end_of_record'){
    if(lf>0&&lh<lf) results.push([lh/lf*100,lh,lf,sf]);
    sf='';lf=0;lh=0;
  }
}
console.log('Total incomplete files:', results.length);
results.sort((a,b)=>a[0]-b[0]);
results.slice(0,5).forEach(r=>console.log(r[0].toFixed(2)+'% ('+r[1]+'/'+r[2]+') '+r[3]));
