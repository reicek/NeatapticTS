/** WORKERS */
export const workers = {
  node: {
    TestWorker: import('./node/testworker'),
  },
  browser: {
    TestWorker: import('./browser/testworker'),
  },
};
