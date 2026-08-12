import { launchLocalServer } from './harness-launcher.js';

const handle = await launchLocalServer({
  cwd: process.cwd(),
  port: Number(process.env.SPAWN_SMOKE_PORT ?? 8090),
  scenarioPath:
    '/docs/browser-tests/scenarios/neatenstein-spawn-at-corners-smoke.html',
});

console.log('SERVER_READY ' + handle.scenarioUrl);

process.on('SIGTERM', async () => {
  await handle.teardown();
  process.exit(0);
});
process.on('SIGINT', async () => {
  await handle.teardown();
  process.exit(0);
});

// Keep alive
setInterval(() => {}, 60000);
