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

  describe('threshold-gated join (implicit AND at or below three tokens)', () => {
    it('keeps the implicit-AND space join at the pinned threshold of three tokens', () => {
      expect(sanitizeFtsQuery('alpha beta gamma')).toBe('alpha* beta* gamma*');
    });

    it('OR-joins the tokens with the uppercase FTS5 keyword at four tokens', () => {
      expect(sanitizeFtsQuery('alpha beta gamma delta')).toBe('alpha* OR beta* OR gamma* OR delta*');
    });

    it('preserves identifier and acronym formatting while OR-joining four tokens', () => {
      expect(sanitizeFtsQuery('network.activate NEAT selection code')).toBe(
        '"network.activate" OR NEAT OR selection* OR code*',
      );
    });

    it('OR-joins a plan-shaped natural-language query so no chunk must carry every token', () => {
      expect(sanitizeFtsQuery('design specification for network activation runtime')).toBe(
        'design* OR specification* OR for* OR network* OR activation* OR runtime*',
      );
    });

    it('OR-joins the canonical plan-shaped fixture query', () => {
      expect(sanitizeFtsQuery('how should the network activation runtime be specified in the plan')).toBe(
        'how* OR should* OR the* OR network* OR activation* OR runtime* OR be* OR specified* OR in* OR the* OR plan*',
      );
    });
  });

  describe('reserved FTS5 keyword tokens', () => {
    it('quotes a user token that is exactly the reserved keyword AND', () => {
      expect(sanitizeFtsQuery('AND')).toBe('"AND"');
    });

    it('quotes a user token that is exactly the reserved keyword OR', () => {
      expect(sanitizeFtsQuery('OR')).toBe('"OR"');
    });

    it('quotes a user token that is exactly the reserved keyword NOT', () => {
      expect(sanitizeFtsQuery('NOT')).toBe('"NOT"');
    });

    it('quotes a reserved keyword user token inside a space-joined query', () => {
      expect(sanitizeFtsQuery('alpha OR beta')).toBe('alpha* "OR" beta*');
    });

    it('keeps a reserved keyword user token quoted when the query is OR-joined', () => {
      expect(sanitizeFtsQuery('alpha OR beta gamma delta')).toBe(
        'alpha* OR "OR" beta* OR gamma* OR delta*',
      );
    });

    it('keeps a lowercase reserved keyword word as a plain word', () => {
      expect(sanitizeFtsQuery('or')).toBe('or*');
    });

    it('does not quote an acronym that merely starts with a reserved keyword', () => {
      expect(sanitizeFtsQuery('ANDROID')).toBe('ANDROID');
    });
  });
});