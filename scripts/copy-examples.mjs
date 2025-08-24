// Copies browser-viewable examples into docs/examples/* so they are published with GitHub Pages.
// Currently only asciiMaze example is needed.
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

try {
  copyAsciiMaze();
} catch (e) {
  console.error('[docs:examples] Failed:', e);
  process.exit(1);
}
