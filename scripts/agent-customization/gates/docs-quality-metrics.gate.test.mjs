/**
 * @module docs-quality-metrics.gate.test
 * @description Direct-import coverage tests for the docs-quality-metrics gate.
 *
 * Runs in the `agent-customization-mjs` Jest project with native ESM so that
 * top-level awaits in the gate and its dependencies are supported. This lets
 * Jest record executable coverage for `docs-quality-metrics.gate.mjs`, which the
 * existing `spawnSync` integration test cannot do.
 */
import assert from 'node:assert/strict';

const gateModule = await import('./docs-quality-metrics.gate.mjs');

describe('docs-quality-metrics.gate.mjs direct import', () => {
  it('runDocsQualityMetricsGate returns a passing mechanism-only report', async () => {
    const report = await gateModule.runDocsQualityMetricsGate();

    assert.equal(report.pass, true);
    assert.equal(report.gateType, 'mechanism-only');
    assert.equal(report.owner, '05-green-testing');
    assert.equal(report.evidence.schema_valid, true);
    assert.equal(report.evidence.deterministic_ordering, true);
    assert.equal(report.evidence.comparator_guards, true);
    assert.equal(report.evidence.parity_cli_mcp, true);
    assert.equal(report.evidence.contract_invalid_rejected, true);
    assert.equal(report.evidence.metric_version, 2);
    assert.equal(report.evidence.scanner_version, '2.0.0');
  });

  it('main --help prints usage and exits cleanly', async () => {
    const logs = [];
    const originalLog = console.log;
    const originalExitCode = process.exitCode;
    console.log = (...args) => {
      logs.push(args.map(String).join(' '));
    };

    try {
      await gateModule.main(['--help']);
      assert.ok(
        logs.some((log) => log.includes('Docs-quality metrics gate')),
        'usage text should mention the gate name',
      );
      assert.equal(process.exitCode, undefined);
    } finally {
      console.log = originalLog;
      process.exitCode = originalExitCode;
    }
  });

  it('main --json prints the report as JSON', async () => {
    const logs = [];
    const originalLog = console.log;
    const originalExitCode = process.exitCode;
    console.log = (...args) => {
      logs.push(args.map(String).join(' '));
    };

    try {
      await gateModule.main(['--json']);
      const report = JSON.parse(logs[0]);
      assert.equal(report.pass, true);
      assert.equal(report.gateType, 'mechanism-only');
      assert.equal(report.owner, '05-green-testing');
    } finally {
      console.log = originalLog;
      process.exitCode = originalExitCode;
    }
  });

  it('main with default argv prints a plain PASS message', async () => {
    const logs = [];
    const originalLog = console.log;
    const originalExitCode = process.exitCode;
    console.log = (...args) => {
      logs.push(args.map(String).join(' '));
    };

    try {
      await gateModule.main();
      assert.ok(
        logs.some((log) => log.includes('PASS docs-quality-metrics gate')),
        'plain output should contain PASS message',
      );
      assert.equal(process.exitCode, 0);
    } finally {
      console.log = originalLog;
      process.exitCode = originalExitCode;
    }
  });

  it('main with no args prints a plain FAIL message and fixHint on failure', async () => {
    const logs = [];
    const originalLog = console.log;
    const originalExitCode = process.exitCode;
    console.log = (...args) => {
      logs.push(args.map(String).join(' '));
    };

    try {
      await gateModule.main([], async () => ({
        pass: false,
        fixHint: 'test failure hint',
      }));
      assert.ok(
        logs.some((log) => log.includes('FAIL docs-quality-metrics gate')),
        'plain output should contain FAIL message',
      );
      assert.ok(
        logs.some((log) => log.includes('test failure hint')),
        'plain output should contain fixHint',
      );
      assert.equal(process.exitCode, 1);
    } finally {
      console.log = originalLog;
      process.exitCode = originalExitCode;
    }
  });

  it('hasMatchingManifestContractFields rejects a missing or invalid right manifest', () => {
    const leftManifest = {
      metricVersion: 2,
      scannerVersion: '2.0.0',
      sourcePathsDigest: 'abc',
      normalizedEvidenceDigest: 'def',
      thresholdConfig: { minJsdocWords: 10, complexityThreshold: 10 },
      scopeConfig: { scopeType: 'paths', scopeDigest: 'ghi', scopeValue: [] },
    };

    assert.equal(
      gateModule.hasMatchingManifestContractFields(leftManifest, null),
      false,
    );
    assert.equal(
      gateModule.hasMatchingManifestContractFields(
        leftManifest,
        'not-an-object',
      ),
      false,
    );
  });

  it('hasMatchingManifestContractFields handles omitted scopeValue arrays', () => {
    const manifest = {
      metricVersion: 2,
      scannerVersion: '2.0.0',
      sourcePathsDigest: 'abc',
      normalizedEvidenceDigest: 'def',
      thresholdConfig: { minJsdocWords: 10, complexityThreshold: 10 },
      scopeConfig: { scopeType: 'paths', scopeDigest: 'ghi' },
    };

    assert.equal(
      gateModule.hasMatchingManifestContractFields(manifest, { ...manifest }),
      true,
    );
  });

  it('buildFixHint returns null on pass and a hint on failure', () => {
    assert.equal(gateModule.buildFixHint(true), null);
    assert.ok(
      typeof gateModule.buildFixHint(false) === 'string' &&
        gateModule.buildFixHint(false).length > 0,
    );
  });
});
