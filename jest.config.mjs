// Pure ESM Jest configuration.
// We avoid CommonJS entirely; Jest 29+ supports loading ESM config (.mjs).

/** @type {import('jest').Config} */
const config = {
  preset: 'ts-jest/presets/default-esm',
  testEnvironment: 'node',
  // Use jsdom for asciiMaze browser example tests (pattern matching)
  testEnvironmentOptions: {},
  projects: [
    {
      displayName: 'default',
      testMatch: [
        '**/src/**/*.test.ts',
        '**/benchmarks/**/*.test.ts',
        '**/examples/**/*.test.ts',
        '**/testing/**/*.test.ts',
      ],
      preset: 'ts-jest/presets/default-esm',
      testEnvironment: 'node',
      extensionsToTreatAsEsm: ['.ts'],
      moduleNameMapper: {
        '^neataptic$': '<rootDir>/src/neataptic.ts',
      },
      transform: {
        '^.+\\.ts$': [
          'ts-jest',
          { useESM: true, tsconfig: 'tsconfig.test.json', diagnostics: true },
        ],
      },
      setupFilesAfterEnv: ['<rootDir>/testing/jest-setup.ts'],
      testTimeout: 300000,
      collectCoverageFrom: [
        'src/**/*.ts',
        '!src/**/*.d.ts',
        '!src/**/*.test.ts',
      ],
      coverageReporters: ['lcov', 'text', 'html', 'json-summary'],
      testPathIgnorePatterns: [
        '/node_modules/',
        '/dist/',
        '/examples/asciiMaze/',
        '/examples/starter-examples.smoke.test.ts',
      ],
    },
    {
      displayName: 'asciiMaze-browser',
      testMatch: ['**/examples/asciiMaze/**/*.test.ts'],
      preset: 'ts-jest/presets/default-esm',
      testEnvironment: 'jsdom',
      extensionsToTreatAsEsm: ['.ts'],
      transform: {
        '^.+\\.ts$': [
          'ts-jest',
          { useESM: true, tsconfig: 'tsconfig.test.json', diagnostics: true },
        ],
      },
      setupFilesAfterEnv: ['<rootDir>/testing/jest-setup.ts'],
      testTimeout: 3000000,
    },
    {
      displayName: 'starter-examples',
      testMatch: ['**/examples/starter-examples.smoke.test.ts'],
      preset: 'ts-jest/presets/default-esm',
      testEnvironment: 'node',
      extensionsToTreatAsEsm: ['.ts'],
      transform: {
        '^.+\\.ts$': [
          'ts-jest',
          { useESM: true, tsconfig: 'tsconfig.test.json', diagnostics: true },
        ],
      },
      setupFilesAfterEnv: ['<rootDir>/testing/jest-setup.ts'],
      testTimeout: 300000,
    },
    {
      displayName: 'semantic-index-scripts',
      testMatch: ['**/scripts/semantic-index/**/*.test.ts'],
      preset: 'ts-jest/presets/default-esm',
      testEnvironment: 'node',
      extensionsToTreatAsEsm: ['.ts'],
      transform: {
        '^.+\\.ts$': [
          'ts-jest',
          { useESM: true, tsconfig: 'tsconfig.test.json', diagnostics: true },
        ],
      },
      setupFilesAfterEnv: ['<rootDir>/testing/jest-setup.ts'],
      testTimeout: 300000,
    },
    {
      displayName: 'agent-customization-scripts',
      testMatch: ['**/scripts/agent-customization/**/*.test.ts'],
      preset: 'ts-jest/presets/default-esm',
      testEnvironment: 'node',
      extensionsToTreatAsEsm: ['.ts'],
      transform: {
        '^.+\\.ts$': [
          'ts-jest',
          { useESM: true, tsconfig: 'tsconfig.test.json', diagnostics: true },
        ],
        '^.+\\.mjs$': [
          '<rootDir>/scripts/agent-customization/mcp/__tests__/mjs-cjs-transformer.cjs',
        ],
      },
      setupFilesAfterEnv: ['<rootDir>/testing/jest-setup.ts'],
      testTimeout: 300000,
      collectCoverageFrom: [
        'scripts/agent-customization/mcp/lazy-facade-core.mjs',
        'scripts/agent-customization/mcp/cortex-facade.mjs',
        'scripts/agent-customization/mcp/devtools-facade.mjs',
      ],
    },
    {
      displayName: 'mcp-semantic-scripts',
      testMatch: ['**/scripts/mcp-semantic/**/*.test.ts'],
      preset: 'ts-jest/presets/default-esm',
      testEnvironment: 'node',
      extensionsToTreatAsEsm: ['.ts'],
      transform: {
        '^.+\\.ts$': [
          'ts-jest',
          { useESM: true, tsconfig: 'tsconfig.test.json', diagnostics: true },
        ],
      },
      setupFilesAfterEnv: ['<rootDir>/testing/jest-setup.ts'],
      testTimeout: 300000,
    },
    {
      displayName: 'semantic-index-mjs',
      testMatch: ['**/scripts/semantic-index/**/*.test.mjs'],
      testEnvironment: 'node',
      transform: {},
      testTimeout: 300000,
    },
    {
      displayName: 'mcp-semantic-mjs',
      testMatch: ['**/scripts/mcp-semantic/**/*.test.mjs'],
      testEnvironment: 'node',
      transform: {},
      testTimeout: 300000,
    },
    {
      displayName: 'analyze-trace-scripts',
      testMatch: ['**/scripts/analyze-trace/**/*.test.ts'],
      preset: 'ts-jest/presets/default-esm',
      testEnvironment: 'node',
      extensionsToTreatAsEsm: ['.ts'],
      moduleNameMapper: {
        '^(\\.{1,2}/.*)\\.js$': '$1',
      },
      transform: {
        '^.+\\.ts$': [
          'ts-jest',
          { useESM: true, tsconfig: 'tsconfig.test.json', diagnostics: true },
        ],
      },
      setupFilesAfterEnv: ['<rootDir>/testing/jest-setup.ts'],
      testTimeout: 300000,
      collectCoverageFrom: ['scripts/analyze-trace/analyze-trace.io.ts'],
    },
    {
      displayName: 'trace-scripts-mjs',
      testMatch: ['**/scripts/trace-*.test.mjs'],
      testEnvironment: 'node',
      transform: {},
      testTimeout: 300000,
      collectCoverageFrom: [
        'scripts/trace-compress.mjs',
        'scripts/trace-summarize.mjs',
      ],
    },
  ],
  testMatch: [
    '**/src/**/*.test.ts',
    '**/benchmarks/**/*.test.ts',
    '**/examples/**/*.test.ts',
    '**/testing/**/*.test.ts',
  ],
  moduleFileExtensions: ['ts', 'js', 'mjs', 'cjs', 'json'],
  extensionsToTreatAsEsm: ['.ts'],
  moduleNameMapper: {
    '^neataptic$': '<rootDir>/src/neataptic.ts',
  },
  transform: {
    '^.+\\.ts$': [
      'ts-jest',
      {
        useESM: true,
        tsconfig: 'tsconfig.test.json',
        diagnostics: true,
      },
    ],
  },
  setupFilesAfterEnv: ['<rootDir>/testing/jest-setup.ts'],
  testTimeout: 300000,
  collectCoverageFrom: [
    'src/**/*.ts',
    '!src/**/*.d.ts',
    '!src/**/*.test.ts',
    'scripts/agent-customization/mcp/lazy-facade-core.mjs',
    'scripts/agent-customization/mcp/cortex-facade.mjs',
    'scripts/agent-customization/mcp/devtools-facade.mjs',
  ],
  coverageReporters: ['lcov', 'text', 'html', 'json-summary'],
  testPathIgnorePatterns: ['/node_modules/', '/dist/'],
};

export default config;
