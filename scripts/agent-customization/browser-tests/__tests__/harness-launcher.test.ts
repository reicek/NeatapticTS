import path from 'node:path';

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..', '..');
const SMOKE_SCENARIO_URL =
  'http://localhost:8080/docs/browser-tests/webgpu-inference-smoke.html';

describe('browser harness launcher', () => {
  describe('launchLocalServer', () => {
    it('starts the local docs server and resolves the smoke scenario URL', async () => {
      const launcher = await import('../harness-launcher');
      const result = await launcher.launchLocalServer({
        cwd: REPO_ROOT,
        port: 8080,
      });

      expect(result).toMatchObject({
        serverUrl: 'http://localhost:8080',
        scenarioUrl: SMOKE_SCENARIO_URL,
      });

      await result.teardown();
    });
  });
});
