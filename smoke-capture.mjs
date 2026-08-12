import puppeteer from 'puppeteer';

const browser = await puppeteer.launch({ headless: true });
const page = await browser.newPage();
await page.goto('http://127.0.0.1:8091/docs/examples/neatenstein/index.html?cachebust=' + Date.now(), {
  waitUntil: 'networkidle0',
});
await new Promise((resolve) => setTimeout(resolve, 3000));
const screenshot = await page.screenshot({ encoding: 'base64' });
await browser.close();
console.log(screenshot);
