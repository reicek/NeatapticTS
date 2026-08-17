export const DOCS_QUALITY_METRIC_VERSION = 2;
export const DOCS_QUALITY_SCANNER_VERSION = '2.0.0';

const REQUIRED_MANIFEST_FIELDS = [
  'metricVersion',
  'scannerVersion',
  'gitCommit',
  'generatedAt',
  'thresholdConfig',
  'scopeConfig',
  'sourcePathsDigest',
  'issueBreakdown',
  'weakCount',
  'normalizedEvidenceDigest',
];

/**
 * Validate the canonical docs-quality manifest v1 contract.
 *
 * @param {unknown} manifest - Candidate manifest payload.
 * @returns {{ valid: boolean, errors: Array<{ field: string, message: string }> }} Validation result.
 */
export function validateDocsQualityManifestV1(manifest) {
  const errors = [];
  if (!isPlainObject(manifest)) {
    return {
      valid: false,
      errors: [{ field: 'manifest', message: 'Manifest must be an object.' }],
    };
  }

  for (const fieldName of REQUIRED_MANIFEST_FIELDS) {
    if (!(fieldName in manifest)) {
      errors.push({
        field: fieldName,
        message: `Missing required field: ${fieldName}`,
      });
    }
  }

  const LEGACY_TOP_LEVEL_FIELDS = ['threshold', 'scopeType', 'scopeDigest'];
  for (const legacyField of LEGACY_TOP_LEVEL_FIELDS) {
    if (Object.hasOwn(manifest, legacyField)) {
      errors.push({
        field: legacyField,
        message: `Legacy top-level field is not allowed: ${legacyField}`,
      });
    }
  }

  if (!Number.isInteger(manifest.metricVersion)) {
    errors.push({
      field: 'metricVersion',
      message: 'metricVersion must be an integer.',
    });
  }

  if (
    typeof manifest.scannerVersion !== 'string' ||
    !manifest.scannerVersion.trim()
  ) {
    errors.push({
      field: 'scannerVersion',
      message: 'scannerVersion must be a non-empty string.',
    });
  }

  if (!isPlainObject(manifest.thresholdConfig)) {
    errors.push({
      field: 'thresholdConfig',
      message: 'thresholdConfig must be an object.',
    });
  }

  if (!isPlainObject(manifest.scopeConfig)) {
    errors.push({
      field: 'scopeConfig',
      message: 'scopeConfig must be an object.',
    });
  }

  if (manifest.generatedAt === '1970-01-01T00:00:00.000Z') {
    errors.push({
      field: 'generatedAt',
      message:
        'generatedAt must be a fresh timestamp, not the frozen canonical value.',
    });
  }

  if (
    typeof manifest.sourcePathsDigest !== 'string' ||
    !manifest.sourcePathsDigest.trim()
  ) {
    errors.push({
      field: 'sourcePathsDigest',
      message: 'sourcePathsDigest must be a non-empty string.',
    });
  }

  if (!isPlainObject(manifest.issueBreakdown)) {
    errors.push({
      field: 'issueBreakdown',
      message: 'issueBreakdown must be an object.',
    });
  }

  if (!Number.isInteger(manifest.weakCount) || manifest.weakCount < 0) {
    errors.push({
      field: 'weakCount',
      message: 'weakCount must be a non-negative integer.',
    });
  }

  if (
    typeof manifest.normalizedEvidenceDigest !== 'string' ||
    !manifest.normalizedEvidenceDigest.trim()
  ) {
    errors.push({
      field: 'normalizedEvidenceDigest',
      message: 'normalizedEvidenceDigest must be a non-empty string.',
    });
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

function isPlainObject(value) {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
