import puppeteer from 'puppeteer';
import fs from 'fs';
const url = process.argv[2] || 'http://127.0.0.1:8091/docs/examples/neatenstein/index.html';
const browser = await puppeteer.launch({ headless: false, args: ['--no-sandbox', '--disable-setuid-sandbox'] });

const page = await browser.newPage();
await page.evaluateOnNewDocument(() => {
  const orig = window.requestAnimationFrame.bind(window);
  window.requestAnimationFrame = (cb) => {
    console.log('rAF scheduled');
    return orig((t) => { console.log('rAF fired'); cb(t); });
  };
});
await page.setViewport({ width: 800, height: 600 });
const consoleLogs = [];
page.on('console', m => consoleLogs.push(`[${m.type()}] ${m.text()}`));
page.on('pageerror', e => consoleLogs.push('PAGEERROR: ' + e.message));
page.on('workercreated', w => {
  console.log('worker created', w.url());
  w.on('console', m => consoleLogs.push(`[worker ${m.type()}] ${m.text()}`));
  w.on('error', e => consoleLogs.push('WORKERERROR: ' + e.message));
});
try {
  await page.goto(url, { waitUntil: 'networkidle0', timeout: 60000 });
  await new Promise(r => setTimeout(r, 5000));
  const metrics = await page.evaluate(() => {
    const canvas = document.querySelector('canvas');
    const rect = canvas.getBoundingClientRect();
    const status = document.getElementById('status');
    return { innerWidth, innerHeight, hasStart: typeof window.neatensteinStart, statusText: status ? status.textContent : null, canvasRect: { x: rect.x, y: rect.y, w: rect.width, h: rect.height }, htmlW: canvas.width, htmlH: canvas.height };
  });
  console.log('metrics', JSON.stringify(metrics));
  const screenshot = await page.screenshot({ type: 'png' });
  fs.writeFileSync('screenshot.png', screenshot);
  // sample center of canvas area for non-black by reading visible canvas into an in-memory 2D context
  const sample = await page.evaluate(() => {
    const canvas = document.querySelector('canvas');
    const rect = canvas.getBoundingClientRect();
    const cx = Math.floor(rect.x + rect.width/2);
    const cy = Math.floor(rect.y + rect.height/2);
    const probe = document.createElement('canvas');
    probe.width = 1; probe.height = 1;
    const pctx = probe.getContext('2d');
    pctx.drawImage(canvas, cx - rect.x, cy - rect.y, 1, 1, 0, 0, 1, 1);
    return { cx, cy, data: Array.from(pctx.getImageData(0,0,1,1).data) };
  });
  console.log('center sample', sample);
  const buf = await page.evaluate(() => {
    const c = document.querySelector('canvas');
    const probe = document.createElement('canvas');
    const w = Math.min(64, c.width || 64), h = Math.min(64, c.height || 64);
    probe.width = w; probe.height = h;
    const pctx = probe.getContext('2d');
    pctx.drawImage(c, 0, 0, w, h);
    return Array.from(pctx.getImageData(0,0,w,h).data).slice(0,16);
  });
  console.log('host context readback', buf);
  console.log('console logs', consoleLogs);
} catch (e) {
  console.error('smoke failed', e);
}
await browser.close();
