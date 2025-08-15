// Suppress console.warn and console.error for tests that intentionally trigger warnings
beforeAll(() => {
  jest.spyOn(console, 'warn').mockImplementation(() => {});
  jest.spyOn(console, 'error').mockImplementation(() => {});
});
afterAll(() => {
  (console.warn as jest.Mock).mockRestore();
  (console.error as jest.Mock).mockRestore();
});
