/**
 * Phase 1 – Field Audit & Slimming Baseline
 * Captures current enumerable own property keys for Connection and Node instances
 * so later slimming / bitfield refactors can assert parity or intentional changes.
 */
import fs from 'fs';
import path from 'path';
import Connection from '../../src/architecture/connection';
import Node from '../../src/architecture/node';

/**
 * Persist field audit data into the shared benchmark results artifact so later
 * slimming refactors can diff property counts & names. Idempotent per test run.
 */
function writeAudit(data: any) {
  const resultsPath = path.resolve(__dirname, 'benchmark.results.json');
  let current: any = {};
  if (fs.existsSync(resultsPath)) {
    try {
      current = JSON.parse(fs.readFileSync(resultsPath, 'utf8'));
    } catch {
      current = {};
    }
  }
  current.fieldAudit = data;
  fs.writeFileSync(resultsPath, JSON.stringify(current, null, 2));
}

describe('phase1.fieldAudit baseline', () => {
  /**
   * Arrange (shared): construct a minimal 3-node chain input -> hidden -> output plus one
   * additional connection input -> hidden returned from connect for auditing. We stimulate
   * activation & propagation to ensure lazily-created fields appear (important for slimming
   * regressions later).
   */
  const nHidden = new Node('hidden');
  const nIn = new Node('input');
  const nOut = new Node('output');
  const [conn] = nIn.connect(nHidden); // first connection (input->hidden)
  nHidden.connect(nOut); // hidden->output
  // Act: Stimulate runtime-created fields
  nIn.activate(0.5);
  nHidden.activate();
  nOut.activate();
  nOut.propagate(0.01, 0.9, true, 0, 0.3);
  // Capture & sort keys (Act - inspection phase)
  const nodeKeys = Object.keys(nHidden).sort();
  const connectionKeys = Object.keys(conn).sort();
  // Persist audit snapshot (Act - side effect)
  writeAudit({
    generatedAt: new Date().toISOString(),
    Node: { count: nodeKeys.length, keys: nodeKeys },
    Connection: { count: connectionKeys.length, keys: connectionKeys },
  });

  describe('Node keys', () => {
    test('non-zero Node key count', () => {
      expect(nodeKeys.length).toBeGreaterThan(0);
    });
    test('Node keys array is sorted', () => {
      expect(nodeKeys.join(',')).toBe([...nodeKeys].sort().join(','));
    });
  });

  describe('Connection keys', () => {
    test('non-zero Connection key count', () => {
      expect(connectionKeys.length).toBeGreaterThan(0);
    });
    test('Connection keys array is sorted', () => {
      expect(connectionKeys.join(',')).toBe(
        [...connectionKeys].sort().join(',')
      );
    });
  });

  describe('persistence', () => {
    test('writes fieldAudit with matching Node count', () => {
      const resultsPath = path.resolve(__dirname, 'benchmark.results.json');
      let parsed: any = {};
      if (fs.existsSync(resultsPath)) {
        try {
          parsed = JSON.parse(fs.readFileSync(resultsPath, 'utf-8'));
        } catch {}
      }
      const ok =
        parsed.fieldAudit &&
        parsed.fieldAudit.Node &&
        parsed.fieldAudit.Node.count === nodeKeys.length;
      expect(ok).toBe(true);
    });
  });
});
