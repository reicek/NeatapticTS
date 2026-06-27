/**
 * Red-phase contracts for NGE lifecycle integration in runtime adaptation.
 *
 * These tests verify that `adaptOnTick` in the racing curriculum's runtime
 * adaptation engine calls `runNgeLifecycle` instead of its standalone random
 * mutation proposal system.  They will fail until Phase 3 Step 11 is
 * implemented.
 *
 * The tests use source-text inspection (the same pattern as
 * `runtime.adaptation.per-car.test.ts`) because the integration target is a
 * structural change inside the module's closure — the standalone proposal
 * functions must be removed and the lifecycle call must be wired in.
 *
 * Single-expect rule enforced throughout.  AAA structure in every test.
 */
import * as fs from 'node:fs';
import * as path from 'node:path';

const SOURCE_FILE = path.resolve(__dirname, 'runtime.adaptation.ts');

function readSourceText(): string {
  return fs.readFileSync(SOURCE_FILE, 'utf-8');
}

describe('runtime.adaptation NGE lifecycle integration', () => {
  describe('runNgeLifecycle import and invocation', () => {
    it('source file imports runNgeLifecycle from the NGE lifecycle module', () => {
      const sourceText = readSourceText();

      const importsRunNgeLifecycle =
        /import\s+[^;]*runNgeLifecycle[^;]*from\s+['"][^'"]*nge-lifecycle['"]/.test(
          sourceText,
        );

      expect(importsRunNgeLifecycle).toBe(true);
    });

    it('adaptOnTick body calls runNgeLifecycle', () => {
      const sourceText = readSourceText();

      // Match a function call: runNgeLifecycle( — not the import statement
      // (imports use braces, not parens after the identifier).
      const callsRunNgeLifecycle = /runNgeLifecycle\s*\(/.test(sourceText);

      expect(callsRunNgeLifecycle).toBe(true);
    });
  });

  describe('standalone random proposal engine is removed', () => {
    it('source file does not declare proposeCandidateOperations', () => {
      const sourceText = readSourceText();

      const declaresProposeCandidateOperations =
        /\bfunction\s+proposeCandidateOperations\b/.test(sourceText);

      expect(declaresProposeCandidateOperations).toBe(false);
    });

    it('source file does not declare resolveStructuralPool', () => {
      const sourceText = readSourceText();

      const declaresResolveStructuralPool =
        /\bfunction\s+resolveStructuralPool\b/.test(sourceText);

      expect(declaresResolveStructuralPool).toBe(false);
    });

    it('source file does not declare applyOperations (standalone mutation applier)', () => {
      const sourceText = readSourceText();

      const declaresApplyOperations = /\bfunction\s+applyOperations\b/.test(
        sourceText,
      );

      expect(declaresApplyOperations).toBe(false);
    });
  });

  describe('lifecycle applyOutcomes used for rollback decisions', () => {
    it('source file references applyOutcomes from the lifecycle result', () => {
      const sourceText = readSourceText();

      const referencesApplyOutcomes = /applyOutcomes/.test(sourceText);

      expect(referencesApplyOutcomes).toBe(true);
    });
  });
});
