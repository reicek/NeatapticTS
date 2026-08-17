import { jest } from '@jest/globals';
import assert from 'node:assert/strict';

jest.unstable_mockModule('node:fs/promises', () => ({
  readFile: jest.fn(),
  writeFile: jest.fn(),
  readdir: jest.fn(),
  glob: jest.fn(),
  stat: jest.fn(),
  mkdir: jest.fn(),
  appendFile: jest.fn(),
  access: jest.fn(),
  constants: { R_OK: 4, W_OK: 2, F_OK: 0 },
  rm: jest.fn(),
  lstat: jest.fn(),
  link: jest.fn(),
  symlink: jest.fn(),
}));

let mockFs;

beforeEach(async () => {
  jest.resetModules();
  mockFs = await import('node:fs/promises');
  jest.clearAllMocks();
});

async function importModule() {
  const origExit = process.exit;
  process.exit = () => {};
  try {
    await import('./update-agent-models.mjs');
  } catch {
    // import errors are safe to ignore
  }
  await new Promise((r) => setTimeout(r, 200));
  process.exit = origExit;
}

function captureConsole() {
  const logs = [];
  const errors = [];
  const origLog = console.log;
  const origError = console.error;
  console.log = (...args) => logs.push(args.join(' '));
  console.error = (...args) => errors.push(args.join(' '));
  return {
    logs, errors,
    restore() { console.log = origLog; console.error = origError; },
  };
}

describe('update-agent-models', () => {
  it('prints help and exits 0 when --help', async () => {
    const origArgv = process.argv;
    const origExit = process.exit;
    process.argv = ['node', 'update-agent-models.mjs', '--help'];
    process.exit = (code) => { throw new Error(`EXIT:${code}`); };
    const cap = captureConsole();

    await importModule();

    process.argv = origArgv;
    process.exit = origExit;
    cap.restore();
    assert.ok(cap.logs.some((l) => l.includes('Update or remove agent model strings')));
  });

  it('errors when --to not provided and not --remove', async () => {
    const origArgv = process.argv;
    const origExit = process.exit;
    process.argv = ['node', 'update-agent-models.mjs', '--from=old'];
    process.exit = (code) => { throw new Error(`EXIT:${code}`); };
    const cap = captureConsole();

    await importModule();

    process.argv = origArgv;
    process.exit = origExit;
    cap.restore();
    assert.ok(cap.errors.some((e) => e.includes('--to=<model-string> is required')));
  });

  it('removes top-level model fields in --remove mode', async () => {
    const origArgv = process.argv;
    const origExit = process.exit;
    process.argv = ['node', 'update-agent-models.mjs', '--remove', '--json'];
    process.exit = (code) => { throw new Error(`EXIT:${code}`); };
    const cap = captureConsole();

    mockFs.readdir.mockResolvedValue([
      { isFile: () => true, name: 'test.agent.md' },
    ]);
    mockFs.readFile.mockResolvedValue('---\nname: test\nmodel: old-model\n---\nbody\n  model: keep-indented\n');
    mockFs.writeFile.mockResolvedValue();

    await importModule();

    process.argv = origArgv;
    process.exit = origExit;
    cap.restore();
    const json = JSON.parse(cap.logs[0]);
    assert.strictEqual(json.mode, 'remove');
    assert.strictEqual(json.totalChanges, 1);
  });

  it('replaces specific model string with --from and --to', async () => {
    const origArgv = process.argv;
    const origExit = process.exit;
    process.argv = ['node', 'update-agent-models.mjs', '--from=old', '--to=new', '--json'];
    process.exit = (code) => { throw new Error(`EXIT:${code}`); };
    const cap = captureConsole();

    mockFs.readdir.mockResolvedValue([
      { isFile: () => true, name: 'test.agent.md' },
    ]);
    mockFs.readFile.mockResolvedValue("---\nname: test\nmodel: 'old'\n---\nbody\n");

    await importModule();

    process.argv = origArgv;
    process.exit = origExit;
    cap.restore();
    const json = JSON.parse(cap.logs[0]);
    assert.strictEqual(json.mode, 'replace');
    assert.strictEqual(json.totalChanges, 1);
    assert.ok(mockFs.writeFile.mock.calls.length > 0);
  });

  it('replaces ALL model strings when --from omitted', async () => {
    const origArgv = process.argv;
    const origExit = process.exit;
    process.argv = ['node', 'update-agent-models.mjs', '--to=new-model', '--json'];
    process.exit = (code) => { throw new Error(`EXIT:${code}`); };
    const cap = captureConsole();

    mockFs.readdir.mockResolvedValue([
      { isFile: () => true, name: 'a.agent.md' },
    ]);
    mockFs.readFile.mockResolvedValue("---\nname: a\nmodel: something\n---\n");

    await importModule();

    process.argv = origArgv;
    process.exit = origExit;
    cap.restore();
    const json = JSON.parse(cap.logs[0]);
    assert.strictEqual(json.totalChanges, 1);
  });

  it('supports --dry-run without writing', async () => {
    const origArgv = process.argv;
    const origExit = process.exit;
    process.argv = ['node', 'update-agent-models.mjs', '--to=new', '--dry-run', '--json'];
    process.exit = (code) => { throw new Error(`EXIT:${code}`); };
    const cap = captureConsole();

    mockFs.readdir.mockResolvedValue([
      { isFile: () => true, name: 'test.agent.md' },
    ]);
    mockFs.readFile.mockResolvedValue("---\nname: test\nmodel: old\n---\n");

    await importModule();

    process.argv = origArgv;
    process.exit = origExit;
    cap.restore();
    assert.strictEqual(mockFs.writeFile.mock.calls.length, 0);
    const json = JSON.parse(cap.logs[0]);
    assert.strictEqual(json.dryRun, true);
  });

  it('skips files with no model field', async () => {
    const origArgv = process.argv;
    const origExit = process.exit;
    process.argv = ['node', 'update-agent-models.mjs', '--to=new', '--json'];
    process.exit = (code) => { throw new Error(`EXIT:${code}`); };
    const cap = captureConsole();

    mockFs.readdir.mockResolvedValue([
      { isFile: () => true, name: 'test.agent.md' },
    ]);
    mockFs.readFile.mockResolvedValue("---\nname: test\n---\nno model here\n");

    await importModule();

    process.argv = origArgv;
    process.exit = origExit;
    cap.restore();
    const json = JSON.parse(cap.logs[0]);
    assert.strictEqual(json.totalChanges, 0);
    assert.strictEqual(json.filesModified, 0);
  });

  it('handles --from as separate arg (not --from=value)', async () => {
    const origArgv = process.argv;
    const origExit = process.exit;
    process.argv = ['node', 'update-agent-models.mjs', '--from', 'old-val', '--to', 'new-val', '--json'];
    process.exit = (code) => { throw new Error(`EXIT:${code}`); };
    const cap = captureConsole();

    mockFs.readdir.mockResolvedValue([
      { isFile: () => true, name: 'test.agent.md' },
    ]);
    mockFs.readFile.mockResolvedValue("---\nname: test\nmodel: old-val\n---\n");

    await importModule();

    process.argv = origArgv;
    process.exit = origExit;
    cap.restore();
    const json = JSON.parse(cap.logs[0]);
    assert.strictEqual(json.totalChanges, 1);
  });

  it('prints text output in non-json mode with changes', async () => {
    const origArgv = process.argv;
    const origExit = process.exit;
    process.argv = ['node', 'update-agent-models.mjs', '--from=old', '--to=new'];
    process.exit = (code) => { throw new Error(`EXIT:${code}`); };
    const cap = captureConsole();

    mockFs.readdir.mockResolvedValue([
      { isFile: () => true, name: 'test.agent.md' },
    ]);
    mockFs.readFile.mockResolvedValue("---\nname: test\nmodel: old\n---\n");
    mockFs.writeFile.mockResolvedValue();

    await importModule();

    process.argv = origArgv;
    process.exit = origExit;
    cap.restore();
    assert.ok(cap.logs.some((l) => l.includes('update-agent-models')));
    assert.ok(cap.logs.some((l) => l.includes('routing-table')));
  });

  it('handles double-quote model values', async () => {
    const origArgv = process.argv;
    const origExit = process.exit;
    process.argv = ['node', 'update-agent-models.mjs', '--to=new', '--json'];
    process.exit = (code) => { throw new Error(`EXIT:${code}`); };
    const cap = captureConsole();

    mockFs.readdir.mockResolvedValue([
      { isFile: () => true, name: 'test.agent.md' },
    ]);
    mockFs.readFile.mockResolvedValue('---\nname: test\nmodel: "old"\n---\n');

    await importModule();

    process.argv = origArgv;
    process.exit = origExit;
    cap.restore();
    const json = JSON.parse(cap.logs[0]);
    assert.strictEqual(json.totalChanges, 1);
  });

  it('handles --dry_run with underscore', async () => {
    const origArgv = process.argv;
    const origExit = process.exit;
    process.argv = ['node', 'update-agent-models.mjs', '--to=new', '--dry_run', '--json'];
    process.exit = (code) => { throw new Error(`EXIT:${code}`); };
    const cap = captureConsole();

    mockFs.readdir.mockResolvedValue([
      { isFile: () => true, name: 'test.agent.md' },
    ]);
    mockFs.readFile.mockResolvedValue("---\nname: test\nmodel: old\n---\n");

    await importModule();

    process.argv = origArgv;
    process.exit = origExit;
    cap.restore();
    assert.strictEqual(mockFs.writeFile.mock.calls.length, 0);
  });

  it('handles empty agent directory', async () => {
    const origArgv = process.argv;
    const origExit = process.exit;
    process.argv = ['node', 'update-agent-models.mjs', '--to=new', '--json'];
    process.exit = (code) => { throw new Error(`EXIT:${code}`); };
    const cap = captureConsole();

    mockFs.readdir.mockResolvedValue([]);

    await importModule();

    process.argv = origArgv;
    process.exit = origExit;
    cap.restore();
    const json = JSON.parse(cap.logs[0]);
    assert.strictEqual(json.filesScanned, 0);
  });

  it('handles --to as last arg without value', async () => {
    const origArgv = process.argv;
    const origExit = process.exit;
    process.argv = ['node', 'update-agent-models.mjs', '--to'];
    process.exit = (code) => { throw new Error(`EXIT:${code}`); };
    const cap = captureConsole();

    mockFs.readdir.mockResolvedValue([]);

    await importModule();

    process.argv = origArgv;
    process.exit = origExit;
    cap.restore();
    assert.ok(cap.errors.some((e) => e.includes('--to=<model-string> is required')));
  });

  it('skips model line already matching target value when --from omitted', async () => {
    const origArgv = process.argv;
    const origExit = process.exit;
    process.argv = ['node', 'update-agent-models.mjs', '--to=new', '--json'];
    process.exit = (code) => { throw new Error(`EXIT:${code}`); };
    const cap = captureConsole();

    mockFs.readdir.mockResolvedValue([
      { isFile: () => true, name: 'test.agent.md' },
    ]);
    mockFs.readFile.mockResolvedValue("---\nname: test\nmodel: new\n---\n");

    await importModule();

    process.argv = origArgv;
    process.exit = origExit;
    cap.restore();
    const json = JSON.parse(cap.logs[0]);
    assert.strictEqual(json.totalChanges, 0);
  });

  it('prints text output with skipped and non-skipped files', async () => {
    const origArgv = process.argv;
    const origExit = process.exit;
    process.argv = ['node', 'update-agent-models.mjs', '--from=old', '--to=new'];
    process.exit = (code) => { throw new Error(`EXIT:${code}`); };
    const cap = captureConsole();

    mockFs.readdir.mockResolvedValue([
      { isFile: () => true, name: 'a.agent.md' },
      { isFile: () => true, name: 'b.agent.md' },
    ]);
    mockFs.readFile.mockResolvedValueOnce("---\nname: a\nmodel: old\n---\n")
      .mockResolvedValueOnce("---\nname: b\n---\n");
    mockFs.writeFile.mockResolvedValue();

    await importModule();

    process.argv = origArgv;
    process.exit = origExit;
    cap.restore();
    assert.ok(cap.logs.some((l) => l.includes('update-agent-models')));
    assert.ok(cap.logs.some((l) => l.includes('a.agent.md')));
  });

  it('prints text output with no changes', async () => {
    const origArgv = process.argv;
    const origExit = process.exit;
    process.argv = ['node', 'update-agent-models.mjs', '--to=new'];
    process.exit = (code) => { throw new Error(`EXIT:${code}`); };
    const cap = captureConsole();

    mockFs.readdir.mockResolvedValue([
      { isFile: () => true, name: 'test.agent.md' },
    ]);
    mockFs.readFile.mockResolvedValue("---\nname: test\n---\nno model\n");

    await importModule();

    process.argv = origArgv;
    process.exit = origExit;
    cap.restore();
    assert.ok(cap.logs.some((l) => l.includes('changes=0')));
    assert.ok(!cap.logs.some((l) => l.includes('routing-table')));
  });

  it('prints text output in dry-run mode with changes', async () => {
    const origArgv = process.argv;
    const origExit = process.exit;
    process.argv = ['node', 'update-agent-models.mjs', '--to=new', '--dry-run'];
    process.exit = (code) => { throw new Error(`EXIT:${code}`); };
    const cap = captureConsole();

    mockFs.readdir.mockResolvedValue([
      { isFile: () => true, name: 'test.agent.md' },
    ]);
    mockFs.readFile.mockResolvedValue("---\nname: test\nmodel: old\n---\n");

    await importModule();

    process.argv = origArgv;
    process.exit = origExit;
    cap.restore();
    assert.ok(cap.logs.some((l) => l.includes('[DRY RUN]')));
    assert.ok(!cap.logs.some((l) => l.includes('routing-table')));
  });
});