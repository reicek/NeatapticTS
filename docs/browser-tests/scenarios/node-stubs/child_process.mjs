// Browser stub for Node child_process built-in.
// The node worker code is never exercised in browser scenarios; this stub
// prevents the externalized esbuild import from failing module load.
export function fork() {
  throw new Error('child_process.fork is not available in the browser');
}
