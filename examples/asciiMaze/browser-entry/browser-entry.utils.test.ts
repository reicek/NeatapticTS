import { createBrowserEvolutionSettings } from './browser-entry.utils';

describe('createBrowserEvolutionSettings', () => {
  it('defaults browser curriculum phases to feed-forward-only structural growth', () => {
    const initialSettings = createBrowserEvolutionSettings(8);

    expect(initialSettings.allowRecurrent).toBe(false);
  });

  it('uses a 100-genome browser population for the opening curriculum phase', () => {
    const initialSettings = createBrowserEvolutionSettings(8);

    expect(initialSettings.popSize).toBe(100);
  });

  it('adds a stronger adaptive-mutation profile for the initial curriculum phase', () => {
    const initialSettings = createBrowserEvolutionSettings(8);

    expect(initialSettings.adaptiveMutation).toEqual({
      enabled: true,
      strategy: 'twoTier',
      adaptEvery: 5,
      sigma: 0.1,
      minRate: 0.001,
    });
  });

  it('leaves the adaptive-mutation override unset for later curriculum phases', () => {
    const laterSettings = createBrowserEvolutionSettings(12);

    expect(laterSettings.adaptiveMutation).toBeUndefined();
  });

  it('keeps later browser curriculum phases feed-forward-only as well', () => {
    const laterSettings = createBrowserEvolutionSettings(12);

    expect(laterSettings.allowRecurrent).toBe(false);
  });

  it('keeps the later browser curriculum population at 100 genomes as well', () => {
    const laterSettings = createBrowserEvolutionSettings(12);

    expect(laterSettings.popSize).toBe(100);
  });
});