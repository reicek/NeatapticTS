/**
 * @module mcp-utils.direct.test
 * @description Native-ESM Jest coverage tests for `mcp-utils.mjs`.
 *
 * Runs in the `agent-customization-mjs` Jest project so V8 instruments the
 * source `.mjs` file directly, producing accurate coverage for shared MCP
 * helpers such as `formatToolResult`.
 */

describe('mcp-utils formatToolResult direct contracts', () => {
  it('serializes a plain object into a JSON text block', async () => {
    const { formatToolResult } = await import('./mcp-utils.mjs');
    const result = formatToolResult({ ok: true, count: 3 });

    expect(result).toEqual({
      isError: false,
      content: [
        {
          type: 'text',
          text: JSON.stringify({ ok: true, count: 3 }, null, 2),
        },
      ],
      structuredContent: { ok: true, count: 3 },
    });
  });

  it('passes through a pre-formatted content array unchanged', async () => {
    const { formatToolResult } = await import('./mcp-utils.mjs');
    const preformatted = {
      isError: false,
      content: [{ type: 'text', text: 'already formatted' }],
      extra: 'preserved',
    };
    const result = formatToolResult(preformatted);

    expect(result).toEqual({
      isError: false,
      content: [{ type: 'text', text: 'already formatted' }],
      extra: 'preserved',
    });
  });

  it('wraps a non-object result as a structured value', async () => {
    const { formatToolResult } = await import('./mcp-utils.mjs');
    const result = formatToolResult('plain text');

    expect(result).toEqual({
      isError: false,
      content: [
        {
          type: 'text',
          text: JSON.stringify({ value: 'plain text' }, null, 2),
        },
      ],
      structuredContent: { value: 'plain text' },
    });
  });

  it('wraps a null result as a null structured value', async () => {
    const { formatToolResult } = await import('./mcp-utils.mjs');
    const result = formatToolResult(null);

    expect(result).toEqual({
      isError: false,
      content: [
        {
          type: 'text',
          text: JSON.stringify({ value: null }, null, 2),
        },
      ],
      structuredContent: { value: null },
    });
  });

  it('wraps an undefined result as a null structured value', async () => {
    const { formatToolResult } = await import('./mcp-utils.mjs');
    const result = formatToolResult(undefined);

    expect(result).toEqual({
      isError: false,
      content: [
        {
          type: 'text',
          text: JSON.stringify({ value: null }, null, 2),
        },
      ],
      structuredContent: { value: null },
    });
  });
});
