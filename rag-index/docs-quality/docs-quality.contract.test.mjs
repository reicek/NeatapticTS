import { validateDocsQualityManifestV1 } from './docs-quality.contract.mjs';

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

describe('docs-quality.contract.mjs', () => {
  describe('validateDocsQualityManifestV1', () => {
    it('returns valid=true for a complete manifest', () => {
      const result = validateDocsQualityManifestV1(createValidManifest());
      expect(result.valid).toBe(true);
      expect(result.errors).toEqual([]);
    });

    it('returns invalid when manifest is not an object', () => {
      const result = validateDocsQualityManifestV1(null);
      expect(result.valid).toBe(false);
      expect(result.errors).toHaveLength(1);
      expect(result.errors[0].field).toBe('manifest');
    });

    it('returns invalid when manifest is an array', () => {
      const result = validateDocsQualityManifestV1([]);
      expect(result.valid).toBe(false);
      expect(result.errors[0].field).toBe('manifest');
    });

    it('returns invalid when manifest is a string', () => {
      const result = validateDocsQualityManifestV1('hello');
      expect(result.valid).toBe(false);
      expect(result.errors[0].field).toBe('manifest');
    });

    it('reports missing required fields', () => {
      const manifest = createValidManifest();
      delete manifest.metricVersion;
      delete manifest.gitCommit;
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.valid).toBe(false);
      const missingFields = result.errors.map((e) => e.field);
      expect(missingFields).toContain('metricVersion');
      expect(missingFields).toContain('gitCommit');
    });

    it('reports all missing fields when manifest is empty object', () => {
      const result = validateDocsQualityManifestV1({});
      expect(result.valid).toBe(false);
      const fields = result.errors.map((e) => e.field);
      expect(fields).toContain('metricVersion');
      expect(fields).toContain('scannerVersion');
      expect(fields).toContain('gitCommit');
      expect(fields).toContain('generatedAt');
      expect(fields).toContain('thresholdConfig');
      expect(fields).toContain('scopeConfig');
      expect(fields).toContain('sourcePathsDigest');
      expect(fields).toContain('issueBreakdown');
      expect(fields).toContain('weakCount');
      expect(fields).toContain('normalizedEvidenceDigest');
    });

    it('rejects legacy top-level fields', () => {
      const manifest = createValidManifest();
      manifest.threshold = 10;
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.valid).toBe(false);
      const fields = result.errors.map((e) => e.field);
      expect(fields).toContain('threshold');
    });

    it('rejects all three legacy fields', () => {
      const manifest = createValidManifest();
      manifest.threshold = 10;
      manifest.scopeType = 'src';
      manifest.scopeDigest = 'abc';
      const result = validateDocsQualityManifestV1(manifest);
      const fields = result.errors.map((e) => e.field);
      expect(fields).toContain('threshold');
      expect(fields).toContain('scopeType');
      expect(fields).toContain('scopeDigest');
    });

    it('reports error when metricVersion is not an integer', () => {
      const manifest = createValidManifest();
      manifest.metricVersion = 2.5;
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.valid).toBe(false);
      const err = result.errors.find((e) => e.field === 'metricVersion');
      expect(err.message).toContain('integer');
    });

    it('reports error when metricVersion is a string', () => {
      const manifest = createValidManifest();
      manifest.metricVersion = 'two';
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.valid).toBe(false);
      expect(result.errors.some((e) => e.field === 'metricVersion')).toBe(true);
    });

    it('reports error when scannerVersion is empty', () => {
      const manifest = createValidManifest();
      manifest.scannerVersion = '  ';
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.valid).toBe(false);
      expect(result.errors.some((e) => e.field === 'scannerVersion')).toBe(true);
    });

    it('reports error when scannerVersion is not a string', () => {
      const manifest = createValidManifest();
      manifest.scannerVersion = 123;
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.valid).toBe(false);
      expect(result.errors.some((e) => e.field === 'scannerVersion')).toBe(true);
    });

    it('reports error when thresholdConfig is not an object', () => {
      const manifest = createValidManifest();
      manifest.thresholdConfig = null;
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.valid).toBe(false);
      expect(result.errors.some((e) => e.field === 'thresholdConfig')).toBe(true);
    });

    it('reports error when thresholdConfig is an array', () => {
      const manifest = createValidManifest();
      manifest.thresholdConfig = [];
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.valid).toBe(false);
      expect(result.errors.some((e) => e.field === 'thresholdConfig')).toBe(true);
    });

    it('reports error when scopeConfig is not an object', () => {
      const manifest = createValidManifest();
      manifest.scopeConfig = 'string';
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.valid).toBe(false);
      expect(result.errors.some((e) => e.field === 'scopeConfig')).toBe(true);
    });

    it('reports error when scopeConfig is null', () => {
      const manifest = createValidManifest();
      manifest.scopeConfig = null;
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.valid).toBe(false);
      expect(result.errors.some((e) => e.field === 'scopeConfig')).toBe(true);
    });

    it('reports error when generatedAt is the frozen canonical value', () => {
      const manifest = createValidManifest();
      manifest.generatedAt = '1970-01-01T00:00:00.000Z';
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.valid).toBe(false);
      expect(result.errors.some((e) => e.field === 'generatedAt')).toBe(true);
    });

    it('reports error when sourcePathsDigest is empty', () => {
      const manifest = createValidManifest();
      manifest.sourcePathsDigest = '';
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.valid).toBe(false);
      expect(result.errors.some((e) => e.field === 'sourcePathsDigest')).toBe(true);
    });

    it('reports error when sourcePathsDigest is not a string', () => {
      const manifest = createValidManifest();
      manifest.sourcePathsDigest = 42;
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.valid).toBe(false);
      expect(result.errors.some((e) => e.field === 'sourcePathsDigest')).toBe(true);
    });

    it('reports error when issueBreakdown is not an object', () => {
      const manifest = createValidManifest();
      manifest.issueBreakdown = null;
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.valid).toBe(false);
      expect(result.errors.some((e) => e.field === 'issueBreakdown')).toBe(true);
    });

    it('reports error when issueBreakdown is an array', () => {
      const manifest = createValidManifest();
      manifest.issueBreakdown = [1, 2];
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.valid).toBe(false);
      expect(result.errors.some((e) => e.field === 'issueBreakdown')).toBe(true);
    });

    it('reports error when weakCount is negative', () => {
      const manifest = createValidManifest();
      manifest.weakCount = -1;
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.valid).toBe(false);
      expect(result.errors.some((e) => e.field === 'weakCount')).toBe(true);
    });

    it('reports error when weakCount is not an integer', () => {
      const manifest = createValidManifest();
      manifest.weakCount = 2.5;
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.valid).toBe(false);
      expect(result.errors.some((e) => e.field === 'weakCount')).toBe(true);
    });

    it('reports error when normalizedEvidenceDigest is empty', () => {
      const manifest = createValidManifest();
      manifest.normalizedEvidenceDigest = '';
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.valid).toBe(false);
      expect(result.errors.some((e) => e.field === 'normalizedEvidenceDigest')).toBe(true);
    });

    it('reports error when normalizedEvidenceDigest is not a string', () => {
      const manifest = createValidManifest();
      manifest.normalizedEvidenceDigest = null;
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.valid).toBe(false);
      expect(result.errors.some((e) => e.field === 'normalizedEvidenceDigest')).toBe(true);
    });

    it('accepts weakCount of 0', () => {
      const manifest = createValidManifest();
      manifest.weakCount = 0;
      const result = validateDocsQualityManifestV1(manifest);
      expect(result.valid).toBe(true);
    });
  });
});