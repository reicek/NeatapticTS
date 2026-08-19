import { jest } from '@jest/globals';
import assert from 'node:assert/strict';

import { jsonToMarkdown } from './json-to-md.mjs';

describe('jsonToMarkdown', () => {
  it('converts a simple string value', () => {
    const result = jsonToMarkdown({ mission: 'Maintain agent usability.' });
    assert.ok(result.includes('## Mission'));
    assert.ok(result.includes('Maintain agent usability.'));
  });

  it('converts array values to bullet lists', () => {
    const result = jsonToMarkdown({
      constraints: ['No session log.', 'Do not edit settings.'],
    });
    assert.ok(result.includes('- No session log.'));
    assert.ok(result.includes('- Do not edit settings.'));
  });

  it('converts snake_case keys to Title Case', () => {
    const result = jsonToMarkdown({ my_key: 'value' });
    assert.ok(result.includes('## My Key'));
  });

  it('converts camelCase keys to Title Case', () => {
    const result = jsonToMarkdown({ myKey: 'value' });
    assert.ok(result.includes('## My Key'));
  });

  it('recursively formats nested objects with bolded keys', () => {
    const result = jsonToMarkdown({ outer: { inner: 'val' } });
    assert.ok(result.includes('**Inner:** val'));
  });

  it('converts numbers to strings', () => {
    const result = jsonToMarkdown({ count: 42 });
    assert.ok(result.includes('42'));
  });

  it('handles multiple top-level keys', () => {
    const result = jsonToMarkdown({ a: '1', b: '2' });
    assert.ok(result.includes('## A'));
    assert.ok(result.includes('## B'));
  });

  it('handles nested objects inside arrays', () => {
    const result = jsonToMarkdown({ items: [{ name: 'foo' }] });
    assert.ok(result.includes('## Items'));
    assert.ok(result.includes('[object Object]'));
  });
});
