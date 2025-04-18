/**
 * Utility class for managing workers in both Node.js and browser environments.
 */
export class Workers {
  /**
   * Loads the Node.js test worker dynamically.
   * @returns {Promise<any>} A promise that resolves to the Node.js TestWorker class.
   */
  static async getNodeTestWorker(): Promise<any> {
    const module = await import('./node/testworker');
    return module.TestWorker;
  }

  /**
   * Loads the browser test worker dynamically.
   * @returns {Promise<any>} A promise that resolves to the browser TestWorker class.
   */
  static async getBrowserTestWorker(): Promise<any> {
    const module = await import('./browser/testworker');
    return module.TestWorker;
  }
}
