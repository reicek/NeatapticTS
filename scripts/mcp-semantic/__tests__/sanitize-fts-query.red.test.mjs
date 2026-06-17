/**
 * @module sanitize-fts-query.red.test
 * @description Regression tests for FTS5 query sanitization.
 *
 * `sanitizeFtsQuery` lives in the cortex-db tool because that is where the
 * SQLite FTS5 MATCH expression is built. Dots and other punctuation in
 * code-specific queries (for example `Network.activate`) are treated by
 * FTS5 as syntax operators, so the sanitizer must turn every non-word,
 * non-whitespace character into a separator before building prefix terms.
 */

const SANITIZE_PATH = '../tools/cortex-db.mjs';

function loadSanitizer() {
  return import(SANITIZE_PATH);
}

describe('sanitizeFtsQuery', () => {
  it('strips dots from code-specific queries such as Network.activate', async () => {
    const { sanitizeFtsQuery } = await loadSanitizer();
    const result = sanitizeFtsQuery('Network.activate implementation');
    expect(result).toBe('Network* activate* implementation*');
  });

  it('removes other FTS5 syntax operators like quotes and parentheses', async () => {
    const { sanitizeFtsQuery } = await loadSanitizer();
    const result = sanitizeFtsQuery('"quoted" (grouped)');
    expect(result).toBe('quoted* grouped*');
  });

  it('collapses multiple separators into a single space', async () => {
    const { sanitizeFtsQuery } = await loadSanitizer();
    const result = sanitizeFtsQuery('a..b--c');
    expect(result).toBe('a* b* c*');
  });

  it('returns an empty string for an empty or whitespace-only query', async () => {
    const { sanitizeFtsQuery } = await loadSanitizer();
    const result = sanitizeFtsQuery('   ');
    expect(result).toBe('');
  });

  it('preserves Unicode letters and digits', async () => {
    const { sanitizeFtsQuery } = await loadSanitizer();
    const result = sanitizeFtsQuery('方法123 test');
    expect(result).toBe('方法123* test*');
  });
});
