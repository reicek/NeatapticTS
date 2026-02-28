// Copies browser-viewable examples into docs/examples/* so they are published with GitHub Pages.
import fs from 'fs';
import path from 'path';

function copyAsciiMaze() {
  const srcDir = path.resolve('test', 'examples', 'asciiMaze');
  if (!fs.existsSync(srcDir)) {
    console.warn(
      '[docs:examples] asciiMaze source directory not found, skipping'
    );
    return;
  }
  const destDir = path.resolve('docs', 'examples', 'asciiMaze');
  fs.mkdirSync(destDir, { recursive: true });
  // Copy index.html only (bundle already built to docs/assets). Could copy other static assets if added later.
  const indexSrc = path.join(srcDir, 'index.html');
  if (fs.existsSync(indexSrc)) {
    fs.copyFileSync(indexSrc, path.join(destDir, 'index.html'));
    console.log('[docs:examples] Copied asciiMaze index.html');
  } else {
    console.warn('[docs:examples] asciiMaze index.html missing');
  }
}

function copyFlappyBird() {
  const srcDir = path.resolve('test', 'examples', 'flappy_bird');
  if (!fs.existsSync(srcDir)) {
    console.warn(
      '[docs:examples] flappy_bird source directory not found, skipping'
    );
    return;
  }
  const destDir = path.resolve('docs', 'examples', 'flappy_bird');
  fs.mkdirSync(destDir, { recursive: true });
  // Copy index.html only (bundle already built to docs/assets).
  const indexSrc = path.join(srcDir, 'index.html');
  if (fs.existsSync(indexSrc)) {
    fs.copyFileSync(indexSrc, path.join(destDir, 'index.html'));
    console.log('[docs:examples] Copied flappy_bird index.html');
  } else {
    console.warn('[docs:examples] flappy_bird index.html missing');
  }
}

function writeExamplesLandingPage() {
  const examplesDir = path.resolve('docs', 'examples');
  fs.mkdirSync(examplesDir, { recursive: true });

  const demoEntries = [
    { dirName: 'asciiMaze', label: 'asciiMaze', title: 'ASCII Maze (NeatapticTS)' },
    {
      dirName: 'flappy_bird',
      label: 'flappy_bird',
      title: 'Flappy Bird (NeatapticTS)',
    },
  ];

  const linksMarkup = demoEntries
    .filter((entry) =>
      fs.existsSync(path.join(examplesDir, entry.dirName, 'index.html'))
    )
    .map(
      (entry) =>
        `<li><a href="./${entry.dirName}/index.html">${entry.title}</a> <span class="demo-path">(examples/${entry.label})</span></li>`
    )
    .join('');

  const html = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Examples • NeatapticTS Docs</title>
    <link rel="stylesheet" href="../assets/theme.css" />
  </head>
  <body>
    <header class="topbar">
      <div class="inner">
        <div class="brand"><a href="../index.html">NeatapticTS</a></div>
        <nav class="main-nav">
          <a href="../index.html">Home</a>
          <a href="../index.html">Docs</a>
          <a href="./index.html" class="active">Examples</a>
          <a href="https://github.com/reicek/NeatapticTS" target="_blank" rel="noopener">GitHub</a>
        </nav>
      </div>
    </header>
    <div class="layout">
      <main class="content">
        <h1>Examples</h1>
        <p>Interactive browser demos built from this repository:</p>
        <ul>${linksMarkup}</ul>
        <footer class="site-footer">Generated from source JSDoc • <a href="https://github.com/reicek/NeatapticTS">GitHub</a></footer>
      </main>
    </div>
  </body>
</html>`;

  fs.writeFileSync(path.join(examplesDir, 'index.html'), html, 'utf8');
  console.log('[docs:examples] Wrote examples landing page');
}

try {
  copyAsciiMaze();
  copyFlappyBird();
  writeExamplesLandingPage();
} catch (e) {
  console.error('[docs:examples] Failed:', e);
  process.exit(1);
}
