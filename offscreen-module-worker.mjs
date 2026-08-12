self.onmessage = (e) => {
  const c = e.data.canvas;
  c.width = 640; c.height = 480;
  const ctx = c.getContext('2d');
  const imgd = ctx.getImageData(0,0,c.width,c.height);
  const d = imgd.data;
  for (let i=0;i<d.length;i+=4) { d[i]=0; d[i+1]=200; d[i+2]=200; d[i+3]=255; }
  ctx.putImageData(imgd,0,0);
  ctx.fillStyle='red';
  ctx.fillRect(0,0,80,80);
  if (typeof ctx.commit === 'function') ctx.commit();
  self.postMessage({ ok: true });
};
