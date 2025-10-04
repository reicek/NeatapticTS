import { copyFileSync } from 'node:fs';
import { join } from 'node:path';

const source = join(process.cwd(), 'tmp', 'original-evolutionEngine.ts');
const target = join(process.cwd(), 'test', 'examples', 'asciiMaze', 'evolutionEngine.ts');

copyFileSync(source, target);

console.log(`Restored ${target} from ${source}`);
