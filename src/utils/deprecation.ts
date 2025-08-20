// One-time deprecation warning utility (removed in Phase 2)
const seen = new Set<string>();
export function onceWarn(key: string, message: string) {
  if (seen.has(key)) return;
  // eslint-disable-next-line no-console
  console.warn(message);
  seen.add(key);
}
