import puppeteer from 'puppeteer';

const browser = await puppeteer.launch({ headless: true });
const page = await browser.newPage();
const result = await page.evaluate(async () => {
  const canvas = document.createElement('canvas');
  canvas.width = 100;
  canvas.height = 100;
  const offscreen = canvas.transferControlToOffscreen();
  const w = new Worker(URL.createObjectURL(new Blob([`
    self.onmessage = (e) => {
      const c = e.data.canvas;
      const ctx = c.getContext('2d');
      if (!ctx) {
        self.postMessage({ ok: false, reason: 'getContext returned null' });
        return;
      }
      ctx.fillStyle = 'rgb(0,200,200)';
      ctx.fillRect(0,0,c.width,c.height);
      self.postMessage({ ok: true, width: c.width, height: c.height });
    };
  `], { type: 'application/javascript' })), { type: 'module' });
  return await new Promise((resolve) => {
    w.onmessage = (e) => resolve(e.data);
    w.postMessage({ canvas: offscreen }, [offscreen]);
  });
});
console.log(JSON.stringify(result, null, 2));
await browser.close();
