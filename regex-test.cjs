const puppeteer = require('puppeteer');
(async () => {
  const browser = await puppeteer.launch({ headless: true, args: ['--no-sandbox', '--disable-setuid-sandbox'] });
  const page = await browser.newPage();
  await page.goto('http://127.0.0.1:8091/examples/neatenstein/index.html', { waitUntil: 'networkidle2', timeout: 60000 });
  await new Promise((r) => setTimeout(r, 2000));
  const res = await page.evaluate(() => {
    const regex = /\/examples\//;
    const path = window.location.pathname;
    return { regexString: regex.toString(), test: regex.test(path), path };
  });
  console.log(JSON.stringify(res, null, 2));
  await browser.close();
})();
