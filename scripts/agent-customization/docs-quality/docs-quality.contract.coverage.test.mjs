import { validateDocsQualityManifestV1 } from '../../../rag-index/docs-quality/docs-quality.contract.mjs';

function createValidManifest() {
  return {
    metricVersion: 2,
    scannerVersion: '2.0.0',
    gitCommit: 'abc123',
    generatedAt: '2024-06-01T12:00:00.000Z',
    thresholdConfig: { minJsdocWords: 10, complexityThreshold: 10 },
    scopeConfig: { scopeType: 'src', scopeValue: ['src'] },
    sourcePathsDigest: 'some-digest-string',
    issueBreakdown: { missingJsdoc: 0, weakJsdoc: 0, highComplexity: 0 },
    weakCount: 0,
    normalizedEvidenceDigest: 'evidence-digest',
  };
}

describe('docs-quality.contract.mjs coverage', () => {
  describe('validateDocsQualityManifestV1 — non-object manifest', () => {
    it('rejects null manifest', () => {
      const result = validateDocsQualityManifestV1(null);
      expect(result.valid).toBe(false);
      expect(result.errors).toHaveLength(1);
      expect(result.errors[0].field).toBe('manifest');
    });

    it('rejects array manifest', () => {
      const result = validateDocsQualityManifestV1([]);
      expect(result.valid).toBe(false);
      expect(result.errors[0].field).toBe('manifest');
    });

    it('rejects string manifest', () => {
      const result = validateDocsQualityManifestV1('hello');
      expect(result.valid).toBe(false);
      expect(result.errors[0].field).toBe('manifest');
    });

    it('rejects number manifest', () => {
      const result = validateDocsQualityManifestV1(42);
      expect(result.valid).toBe(false);
    });
  });

  describe('validateDocsQualityManifestV1 — legacy fields', () => {
    it('rejects legacy threshold field', () => {
      const manifest = createValidManifest();
      manifest.threshold = 10;
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.valid).toBe(false);
      expect(result.errors.some((e) => e.field === 'threshold')).toBe(true);
    });

    it('rejects legacy scopeType field', () => {
      const manifest = createValidManifest();
      manifest.scopeType = 'src';
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.errors.some((e) => e.field === 'scopeType')).toBe(true);
    });

    it('rejects legacy scopeDigest field', () => {
      const manifest = createValidManifest();
      manifest.scopeDigest = 'abc';
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.errors.some((e) => e.field === 'scopeDigest')).toBe(true);
    });
  });

  describe('validateDocsQualityManifestV1 — metricVersion', () => {
    it('rejects non-integer metricVersion (float)', () => {
      const manifest = createValidManifest();
      manifest.metricVersion = 2.5;
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.errors.some((e) => e.field === 'metricVersion')).toBe(true);
    });

    it('rejects non-integer metricVersion (string)', () => {
      const manifest = createValidManifest();
      manifest.metricVersion = 'two';
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.errors.some((e) => e.field === 'metricVersion')).toBe(true);
    });
  });

  describe('validateDocsQualityManifestV1 — scannerVersion', () => {
    it('rejects empty scannerVersion', () => {
      const manifest = createValidManifest();
      manifest.scannerVersion = '  ';
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.errors.some((e) => e.field === 'scannerVersion')).toBe(true);
    });

    it('rejects non-string scannerVersion', () => {
      const manifest = createValidManifest();
      manifest.scannerVersion = 123;
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.errors.some((e) => e.field === 'scannerVersion')).toBe(true);
    });
  });

  describe('validateDocsQualityManifestV1 — thresholdConfig', () => {
    it('rejects null thresholdConfig', () => {
      const manifest = createValidManifest();
      manifest.thresholdConfig = null;
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.errors.some((e) => e.field === 'thresholdConfig')).toBe(true);
    });

    it('rejects array thresholdConfig', () => {
      const manifest = createValidManifest();
      manifest.thresholdConfig = [];
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.errors.some((e) => e.field === 'thresholdConfig')).toBe(true);
    });
  });

  describe('validateDocsQualityManifestV1 — scopeConfig', () => {
    it('rejects string scopeConfig', () => {
      const manifest = createValidManifest();
      manifest.scopeConfig = 'string';
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.errors.some((e) => e.field === 'scopeConfig')).toBe(true);
    });

    it('rejects null scopeConfig', () => {
      const manifest = createValidManifest();
      manifest.scopeConfig = null;
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.errors.some((e) => e.field === 'scopeConfig')).toBe(true);
    });
  });

  describe('validateDocsQualityManifestV1 — generatedAt frozen value', () => {
    it('rejects frozen canonical timestamp', () => {
      const manifest = createValidManifest();
      manifest.generatedAt = '1970-01-01T00:00:00.000Z';
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.errors.some((e) => e.field === 'generatedAt')).toBe(true);
    });
  });

  describe('validateDocsQualityManifestV1 — sourcePathsDigest', () => {
    it('rejects empty sourcePathsDigest', () => {
      const manifest = createValidManifest();
      manifest.sourcePathsDigest = '';
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.errors.some((e) => e.field === 'sourcePathsDigest')).toBe(true);
    });

    it('rejects non-string sourcePathsDigest', () => {
      const manifest = createValidManifest();
      manifest.sourcePathsDigest = 42;
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.errors.some((e) => e.field === 'sourcePathsDigest')).toBe(true);
    });
  });

  describe('validateDocsQualityManifestV1 — issueBreakdown', () => {
    it('rejects null issueBreakdown', () => {
      const manifest = createValidManifest();
      manifest.issueBreakdown = null;
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.errors.some((e) => e.field === 'issueBreakdown')).toBe(true);
    });

    it('rejects array issueBreakdown', () => {
      const manifest = createValidManifest();
      manifest.issueBreakdown = [1, 2];
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.errors.some((e) => e.field === 'issueBreakdown')).toBe(true);
    });
  });

  describe('validateDocsQualityManifestV1 — weakCount', () => {
    it('rejects negative weakCount', () => {
      const manifest = createValidManifest();
      manifest.weakCount = -1;
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.errors.some((e) => e.field === 'weakCount')).toBe(true);
    });

    it('rejects non-integer weakCount', () => {
      const manifest = createValidManifest();
      manifest.weakCount = 2.5;
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.errors.some((e) => e.field === 'weakCount')).toBe(true);
    });
  });

  describe('validateDocsQualityManifestV1 — normalizedEvidenceDigest', () => {
    it('rejects empty normalizedEvidenceDigest', () => {
      const manifest = createValidManifest();
      manifest.normalizedEvidenceDigest = '';
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.errors.some((e) => e.field === 'normalizedEvidenceDigest')).toBe(true);
    });

    it('rejects non-string normalizedEvidenceDigest', () => {
      const manifest = createValidManifest();
      manifest.normalizedEvidenceDigest = null;
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.errors.some((e) => e.field === 'normalizedEvidenceDigest')).toBe(true);
    });
  });

  describe('validateDocsQualityManifestV1 — valid manifest', () => {
    it('accepts a complete valid manifest', () => {
      const result = validateDocsQualityManifestV1(createValidManifest());
      expect(result.valid).toBe(true);
      expect(result.errors).toEqual([]);
    });

    it('accepts weakCount of 0', () => {
      const manifest = createValidManifest();
      manifest.weakCount = 0;
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.valid).toBe(true);
    });
  });
});