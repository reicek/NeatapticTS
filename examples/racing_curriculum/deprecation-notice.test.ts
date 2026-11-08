import { readFileSync } from 'node:fs';
import { join } from 'node:path';

/**
 * AC-039 source-text inspection: both the racing_curriculum README and the
 * examples README must carry a deprecation notice pointing to
 * `examples/neatRacing` as the successor. The racing_curriculum folder itself
 * is NOT removed and no wrapper code is introduced.
 */
const REPO_ROOT = process.cwd();
const RACING_README_PATH = join(REPO_ROOT, 'examples', 'racing_curriculum', 'README.md');
const EXAMPLES_README_PATH = join(REPO_ROOT, 'examples', 'README.md');

describe('racing_curriculum deprecation notices (AC-039)', () => {
  it('racing_curriculum README carries the deprecation notice', () => {
    const readme = readFileSync(RACING_README_PATH, 'utf8');
    expect(readme).toContain('> **Deprecated:**');
  });

  it('racing_curriculum README names examples/neatRacing as the successor', () => {
    const readme = readFileSync(RACING_README_PATH, 'utf8');
    expect(readme).toContain('examples/neatRacing');
  });

  it('examples README carries the racing_curriculum deprecation notice', () => {
    const readme = readFileSync(EXAMPLES_README_PATH, 'utf8');
    expect(readme).toContain('**Status: deprecated.**');
  });

  it('examples README names examples/neatRacing as the successor', () => {
    const readme = readFileSync(EXAMPLES_README_PATH, 'utf8');
    expect(readme).toContain('examples/neatRacing');
  });
});
