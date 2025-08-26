// Minimal ambient declarations to allow ES2023 Array methods to be used in tests.
// This is a scoped test-only declaration file to avoid changing project tsconfig.

interface Array<T> {
  toSorted(compareFn?: (a: T, b: T) => number): T[];
  toReversed?(): T[];
  toSpliced?<T>(start: number, deleteCount?: number, ...items: T[]): T[];
}

declare function structuredClone<T>(value: T): T;
