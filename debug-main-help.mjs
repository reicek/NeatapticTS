import process from 'node:process';
import path from 'node:path';

const REPO_ROOT = path.resolve(process.cwd());
const SERVER_PATH = new URL(
  'file:///' + path.resolve(REPO_ROOT, 'scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs').replace(/\\/g, '/'),
);

process.argv = ['node', 'script', '--help'];
const mod = await import(SERVER_PATH);
const originalExit = process.exit;
process.exit = (code) => {
  console.log('process.exit called with code', code);
  throw new Error('EXIT_THROWN');
};
try {
  await mod.main();
} catch (error) {
  console.log('caught:', error.message);
} finally {
  process.exit = originalExit;
}
