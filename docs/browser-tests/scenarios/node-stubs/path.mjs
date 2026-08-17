// Browser stub for Node path built-in.
// Only exported because the externalized esbuild import requires it;
// the node worker path is not executed in browser scenarios.
export function join(...segments) {
  return segments.join('/');
}
