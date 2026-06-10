/**
 * @module build-term-index.red.test
 * @description Red tests for the term index builder pipeline.
 *
 * Tests cover:
 * - extractQualifyingTerms() frequency filter (min 5, max 30%)
 * - extractQualifyingTerms() length filter (min 3 chars)
 * - extractQualifyingTerms() ASCII filter (must contain ASCII letter)
 * - porterTokenize() tokenization and stemming
 * - applyPorterStem() basic stemming rules
 * - Default constants verification
 *
 * Pure .mjs test — runs directly via Jest ESM project (no ts-jest).
 */

// ---------------------------------------------------------------------------
// Helpers — mock database for extractQualifyingTerms tests
// ---------------------------------------------------------------------------

/**
 * Create a mock corpus database that returns the given chunk rows.
 *
 * Each row has chunk_id, heading_path, body_text, and doc_family.
 * The mock only intercepts the specific SQL query that
 * extractQualifyingTerms uses.
 *
 * @param {Array<object>} rows - Chunk rows to return.
 * @returns {{ prepare: Function }} Mock database object.
 */
function createMockDatabase(rows) {
  return {
    prepare() {
      return {
        all() {
          return rows;
        },
      };
    },
  };
}

/**
 * Generate N chunk rows where a specific term appears in each row's heading.
 * Each row gets a unique chunk_id and a simple doc_family.
 *
 * @param {number} count - Number of rows to generate.
 * @param {string} term - The term to embed in heading_path.
 * @param {number} [startId=0] - Starting chunk_id.
 * @returns {Array<object>} Generated chunk rows.
 */
function generateRowsWithTerm(count, term, startId = 0) {
  return Array.from({ length: count }, (_, i) => ({
    chunk_id: startId + i,
    heading_path: `# Section with ${term}`,
    body_text: `Body text containing ${term} for chunk ${startId + i}.`,
    doc_family: 'test-family',
  }));
}

// ---------------------------------------------------------------------------
// extractQualifyingTerms: frequency filter
// ---------------------------------------------------------------------------

describe('build-term-index: extractQualifyingTerms frequency filter', () => {
  it('excludes terms appearing fewer than minFrequency times', async () => {
    const { extractQualifyingTerms } = await import('../build-term-index.mjs');

    // "rareterm" appears only 2 times — below minFrequency default of 5
    const rows = [
      ...generateRowsWithTerm(2, 'rareterm', 0),
      ...generateRowsWithTerm(8, 'communterm', 2),
    ];

    const { qualifyingTerms } = extractQualifyingTerms(
      createMockDatabase(rows),
      { minFrequency: 5 },
    );

    expect(qualifyingTerms.has('rareterm')).toBe(false);
  });

  it('excludes terms exceeding maxFrequencyRatio of total chunks', async () => {
    const { extractQualifyingTerms } = await import('../build-term-index.mjs');

    // "stopword" appears in all 10 rows (100%), exceeding 30% threshold
    const rows = generateRowsWithTerm(10, 'stopword', 0);

    const { qualifyingTerms } = extractQualifyingTerms(
      createMockDatabase(rows),
      { maxFrequencyRatio: 0.3 },
    );

    expect(qualifyingTerms.has('stopword')).toBe(false);
  });

  it('includes terms within the frequency window', async () => {
    const { extractQualifyingTerms } = await import('../build-term-index.mjs');

    // "validterm" appears in 6 out of 20 rows (30%), within threshold
    const rows = [
      ...generateRowsWithTerm(6, 'validterm', 0),
      ...generateRowsWithTerm(14, 'otherterm', 6),
    ];

    const { qualifyingTerms } = extractQualifyingTerms(
      createMockDatabase(rows),
      { minFrequency: 5, maxFrequencyRatio: 0.35 },
    );

    expect(qualifyingTerms.has('validterm')).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// extractQualifyingTerms: length filter
// ---------------------------------------------------------------------------

describe('build-term-index: extractQualifyingTerms length filter', () => {
  it('excludes terms shorter than minTermLength (default 3)', async () => {
    const { extractQualifyingTerms } = await import('../build-term-index.mjs');

    // "ab" is 2 chars after stemming, below minTermLength default of 3
    const rows = generateRowsWithTerm(6, 'ab xy validterm', 0);

    const { qualifyingTerms } = extractQualifyingTerms(
      createMockDatabase(rows),
      { minFrequency: 1, maxFrequencyRatio: 1.0, minTermLength: 3 },
    );

    expect(qualifyingTerms.has('ab')).toBe(false);
  });

  it('includes terms meeting the minTermLength threshold', async () => {
    const { extractQualifyingTerms } = await import('../build-term-index.mjs');

    const rows = generateRowsWithTerm(6, 'validterm', 0);

    const { qualifyingTerms } = extractQualifyingTerms(
      createMockDatabase(rows),
      { minFrequency: 1, maxFrequencyRatio: 1.0, minTermLength: 3 },
    );

    expect(qualifyingTerms.has('validterm')).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// extractQualifyingTerms: ASCII filter
// ---------------------------------------------------------------------------

describe('build-term-index: extractQualifyingTerms ASCII filter', () => {
  it('excludes pure numeric tokens that contain no ASCII letters', async () => {
    const { extractQualifyingTerms } = await import('../build-term-index.mjs');

    // "123456" is a pure numeric token — no ASCII letter
    const rows = generateRowsWithTerm(6, '123456 validterm', 0);

    const { qualifyingTerms } = extractQualifyingTerms(
      createMockDatabase(rows),
      { minFrequency: 1, maxFrequencyRatio: 1.0 },
    );

    expect(qualifyingTerms.has('123456')).toBe(false);
  });

  it('includes tokens that contain at least one ASCII letter', async () => {
    const { extractQualifyingTerms } = await import('../build-term-index.mjs');

    const rows = generateRowsWithTerm(6, 'abc123', 0);

    const { qualifyingTerms } = extractQualifyingTerms(
      createMockDatabase(rows),
      { minFrequency: 1, maxFrequencyRatio: 1.0 },
    );

    expect(qualifyingTerms.has('abc123')).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// porterTokenize
// ---------------------------------------------------------------------------

describe('build-term-index: porterTokenize', () => {
  it('lowercases and splits text on non-alphanumeric characters', async () => {
    const { porterTokenize } = await import('../build-term-index.mjs');

    const tokens = porterTokenize('Hello World! Foo-Bar');
    expect(tokens).toContain('hello');
  });

  it('applies Porter stemming to each token', async () => {
    const { porterTokenize } = await import('../build-term-index.mjs');

    // "networks" should be stemmed to "network" (removing trailing 's')
    const tokens = porterTokenize('networks are running');
    expect(tokens).toContain('network');
  });
});

// ---------------------------------------------------------------------------
// applyPorterStem
// ---------------------------------------------------------------------------

describe('build-term-index: applyPorterStem', () => {
  it('returns short words unchanged (length < 4)', async () => {
    const { applyPorterStem } = await import('../build-term-index.mjs');

    expect(applyPorterStem('the')).toBe('the');
  });

  it('returns two-letter words unchanged', async () => {
    const { applyPorterStem } = await import('../build-term-index.mjs');

    expect(applyPorterStem('is')).toBe('is');
  });

  it('strips sses suffix', async () => {
    const { applyPorterStem } = await import('../build-term-index.mjs');

    expect(applyPorterStem('classes')).toBe('class');
  });

  it('strips ies suffix', async () => {
    const { applyPorterStem } = await import('../build-term-index.mjs');

    expect(applyPorterStem('dies')).toBe('di');
  });

  it('preserves ss endings', async () => {
    const { applyPorterStem } = await import('../build-term-index.mjs');

    expect(applyPorterStem('stress')).toBe('stress');
  });

  it('preserves us endings', async () => {
    const { applyPorterStem } = await import('../build-term-index.mjs');

    expect(applyPorterStem('status')).toBe('status');
  });

  it('strips trailing s for plural words', async () => {
    const { applyPorterStem } = await import('../build-term-index.mjs');

    expect(applyPorterStem('dogs')).toBe('dog');
  });

  it('strips eed suffix for words longer than 4 chars', async () => {
    const { applyPorterStem } = await import('../build-term-index.mjs');

    // "agreed" (6 chars) ends with "eed" → slice(0, -1) → "agree"
    expect(applyPorterStem('agreed')).toBe('agree');
  });

  it('strips ed suffix with vowel in stem', async () => {
    const { applyPorterStem } = await import('../build-term-index.mjs');

    expect(applyPorterStem('walked')).toBe('walk');
  });

  it('strips ing suffix with vowel in stem returning runn', async () => {
    const { applyPorterStem } = await import('../build-term-index.mjs');

    // "running" → stem "runn" (has vowel 'u') → ends with 'n' not 'd'/'l' → "runn"
    expect(applyPorterStem('running')).toBe('runn');
  });

  it('strips ing suffix from playing', async () => {
    const { applyPorterStem } = await import('../build-term-index.mjs');

    expect(applyPorterStem('playing')).toBe('play');
  });

  it('strips ational suffix', async () => {
    const { applyPorterStem } = await import('../build-term-index.mjs');

    expect(applyPorterStem('relational')).toBe('relate');
  });

  it('strips tional suffix', async () => {
    const { applyPorterStem } = await import('../build-term-index.mjs');

    // "conditional" → slice(0, -4) + 'e' = "conditi" + 'e' = "conditie"
    expect(applyPorterStem('conditional')).toBe('conditie');
  });

  it('strips ization suffix', async () => {
    const { applyPorterStem } = await import('../build-term-index.mjs');

    expect(applyPorterStem('realization')).toBe('realize');
  });

  it('strips ation suffix', async () => {
    const { applyPorterStem } = await import('../build-term-index.mjs');

    // "creation" → slice(0, -3) = "creat"
    expect(applyPorterStem('creation')).toBe('creat');
  });

  it('preserves happiness because ss check fires before ness', async () => {
    const { applyPorterStem } = await import('../build-term-index.mjs');

    // "happiness" ends with "ss" so the ss rule fires before ness rule
    expect(applyPorterStem('happiness')).toBe('happiness');
  });

  it('strips ment suffix', async () => {
    const { applyPorterStem } = await import('../build-term-index.mjs');

    expect(applyPorterStem('development')).toBe('develop');
  });
});

// ---------------------------------------------------------------------------
// Default constants
// ---------------------------------------------------------------------------

describe('build-term-index: default constants', () => {
  it('exports DEFAULT_MIN_FREQUENCY as 5', async () => {
    const { DEFAULT_MIN_FREQUENCY } = await import('../build-term-index.mjs');

    expect(DEFAULT_MIN_FREQUENCY).toBe(5);
  });

  it('exports DEFAULT_MAX_FREQUENCY_RATIO as 0.3', async () => {
    const { DEFAULT_MAX_FREQUENCY_RATIO } =
      await import('../build-term-index.mjs');

    expect(DEFAULT_MAX_FREQUENCY_RATIO).toBe(0.3);
  });

  it('exports DEFAULT_MIN_TERM_LENGTH as 3', async () => {
    const { DEFAULT_MIN_TERM_LENGTH } = await import('../build-term-index.mjs');

    expect(DEFAULT_MIN_TERM_LENGTH).toBe(3);
  });
});
