import puppeteer from 'puppeteer';
import { readFileSync } from 'node:fs';

const browser = await puppeteer.launch();
const page = await browser.newPage();
const png = readFileSync('C:/Users/reice/.copilot/session-state/0d5d436c-2ec4-4890-a796-6067b582f32f/files/neatenstein-debug.png').toString('base64');
const result = await page.evaluate(async (b64) => {
  const img = new Image();
  img.src = 'data:image/png;base64,' + b64;
  await new Promise((r, x) => (img.onload = r, img.onerror = x));
  const c = document.createElement('canvas');
  c.width = img.naturalWidth;
  c.height = img.naturalHeight;
  const ctx = c.getContext('2d');
  ctx.drawImage(img, 0, 0);
  const d = ctx.getImageData(0, 0, c.width, c.height).data;
  let nonBlack = 0, bright = 0, total = d.length / 4;
  const samples = [];
  for (let y = 0; y < 40; y += 5) {
    for (let x = 0; x < 40; x += 5) {
      const i = (y * c.width + x) * 4;
      samples.push({ x, y, r: d[i], g: d[i+1], b: d[i+2] });
    }
  }
  for (let i = 0; i < d.length; i += 4) {
    const v = Math.max(d[i], d[i+1], d[i+2]);
    if (v > 30) nonBlack += 1;
    if (v > 120) bright += 1;
  }
  return { total, nonBlack, bright, width: c.width, height: c.height, samples };
}, png);
console.log(JSON.stringify(result, null, 2));
await browser.close();
