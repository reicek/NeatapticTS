import { sanitizeFtsQuery } from './tokenizer.mjs';

describe('sanitizeFtsQuery', () => {
  it('returns empty string for null input', () => {
    expect(sanitizeFtsQuery(null)).toBe('');
  });

  it('returns empty string for undefined input', () => {
    expect(sanitizeFtsQuery(undefined)).toBe('');
  });

  it('returns empty string for empty string', () => {
    expect(sanitizeFtsQuery('')).toBe('');
  });

  it('returns empty string for whitespace-only string', () => {
    expect(sanitizeFtsQuery('   ')).toBe('');
  });

  it('returns empty string for punctuation-only string', () => {
    expect(sanitizeFtsQuery('!!! ??? ...')).toBe('');
  });

  it('quotes dotted identifiers', () => {
    expect(sanitizeFtsQuery('network.activate')).toBe('"network.activate"');
  });

  it('quotes snake_case identifiers', () => {
    expect(sanitizeFtsQuery('snake_case_function')).toBe('"snake_case_function"');
  });

  it('quotes camelCase identifiers', () => {
    expect(sanitizeFtsQuery('findTheNEATSelectionCode')).toBe('"findTheNEATSelectionCode"');
  });

  it('keeps acronyms as exact terms', () => {
    expect(sanitizeFtsQuery('NEAT')).toBe('NEAT');
  });

  it('keeps acronyms with numbers as exact terms', () => {
    expect(sanitizeFtsQuery('API2')).toBe('API2');
  });

  it('adds prefix wildcard to plain words', () => {
    expect(sanitizeFtsQuery('selection')).toBe('selection*');
  });

  it('handles mixed tokens', () => {
    expect(sanitizeFtsQuery('NEAT selection code')).toBe('NEAT selection* code*');
  });

  it('handles mixed identifiers and plain words', () => {
    expect(sanitizeFtsQuery('network.activate selection')).toBe('"network.activate" selection*');
  });

  it('handles numeric tokens', () => {
    expect(sanitizeFtsQuery('test123')).toBe('test123*');
  });

  it('handles underscore-only after letter', () => {
    expect(sanitizeFtsQuery('_private')).toBe('_private*');
  });

  it('handles file extension-like tokens', () => {
    expect(sanitizeFtsQuery('code.ts')).toBe('"code.ts"');
  });

  it('handles non-ASCII Unicode letters', () => {
    expect(sanitizeFtsQuery('café')).toBe('café*');
  });

  it('handles mixed scripts', () => {
    const result = sanitizeFtsQuery('NEAT café network.activate');
    expect(result).toContain('"network.activate"');
    expect(result).toContain('NEAT');
    expect(result).toContain('café*');
  });

  it('handles number input (coerced to string)', () => {
    expect(sanitizeFtsQuery(123)).toBe('123*');
  });

  it('handles string with mixed case camelCase', () => {
    expect(sanitizeFtsQuery('myVarName')).toBe('"myVarName"');
  });

  it('handles tokens separated by various punctuation', () => {
    expect(sanitizeFtsQuery('foo;bar')).toBe('foo* bar*');
  });

  it('handles single lowercase letter', () => {
    expect(sanitizeFtsQuery('x')).toBe('x*');
  });

  it('handles single uppercase letter (not acronym)', () => {
    // Single uppercase letter doesn't match ACRONYM_PATTERN (requires 2+ chars)
    expect(sanitizeFtsQuery('X')).toBe('X*');
  });
});