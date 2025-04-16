/** WORKERS */
export class Workers {
  static async getNodeTestWorker() {
    const module = await import('./node/testworker.js');
    return module.TestWorker;
  }

  static async getBrowserTestWorker() {
    const module = await import('./browser/testworker.js');
    return module.TestWorker;
  }
}
