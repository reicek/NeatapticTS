import { pollUntil, PollUntilAbortError } from './pollUntil';

describe('pollUntil', () => {
  it('resolves when predicate becomes truthy', async () => {
    let count = 0;
    const result = await pollUntil(
      () => {
        count++;
        return count > 2 ? 'done' : '';
      },
      { intervalMs: 5, timeoutMs: 200 }
    );
    expect(result).toBe('done');
  });

  it('rejects on timeout', async () => {
    await expect(
      pollUntil(() => false, { intervalMs: 5, timeoutMs: 40 })
    ).rejects.toThrow(/timeout/i);
  });

  it('aborts via signal', async () => {
    const controller = new AbortController();
    setTimeout(() => controller.abort(), 20);
    await expect(
      pollUntil(() => false, {
        intervalMs: 5,
        timeoutMs: 200,
        signal: controller.signal,
      })
    ).rejects.toThrow(/aborted/i);
  });

  it('handles predicate throw', async () => {
    await expect(
      pollUntil(
        () => {
          throw new Error('boom');
        },
        { intervalMs: 5, timeoutMs: 50 }
      )
    ).rejects.toThrow('boom');
  });
});
