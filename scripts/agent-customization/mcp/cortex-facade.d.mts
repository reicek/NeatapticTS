/**
 * Lazy-load MCP facade for the Repo Cortex server.
 */

/**
 * Facade handle returned by `createCortexFacade`.
 */
export interface McpFacade {
  /**
   * Dispatch a JSON-RPC request and return a JSON-RPC response envelope.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  dispatch(request: any): Promise<any>;

  /**
   * Close the underlying child transport.
   */
  close(): Promise<void>;
}

/**
 * Create a lazy-load facade for the Repo Cortex MCP server.
 */
export function createCortexFacade(
  options?: Record<string, unknown>,
): McpFacade;
