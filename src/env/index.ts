/**
 * Default environment adapter entry for repo builds and Node-oriented tests.
 *
 * Browser-target bundlers should alias this module to the browser adapter so
 * Node-only worker code is compiled out at build time rather than selected at
 * runtime.
 */
export { getBrowserTestWorker, getNodeTestWorker } from './node/worker-loader';
