import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';

const mapPath = join(process.cwd(), 'docs', 'assets', 'ascii-maze.bundle.js.map');
const outPath = join(process.cwd(), 'tmp', 'original-evolutionEngine.ts');

const mapRaw = readFileSync(mapPath, 'utf8');
const mapData = JSON.parse(mapRaw);

const targetPath = 'test/examples/asciiMaze/evolutionEngine.ts';
const sourceIndex = mapData.sources.findIndex((source) => source.endsWith(targetPath));
if (sourceIndex === -1) {
  throw new Error(`Could not locate ${targetPath} in sources array.`);
}

const sourceContent = mapData.sourcesContent?.[sourceIndex];
if (typeof sourceContent !== 'string') {
  throw new Error(`sourcesContent for ${targetPath} missing or not a string.`);
}

mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, sourceContent.replace(/\r\n/g, '\n'), 'utf8');

console.log(`Extracted ${targetPath} to ${outPath}`);
