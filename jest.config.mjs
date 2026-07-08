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
        '^.+assimilate-repo\\.mjs$': [
          '<rootDir>/scripts/agent-customization/mcp/__tests__/mjs-cjs-transformer.cjs',
        ],
      },
      setupFilesAfterEnv: ['<rootDir>/testing/jest-setup.ts'],
      testTimeout: 300000,
      collectCoverageFrom: [
        'src/**/*.ts',
        '!src/**/*.d.ts',
        '!src/**/*.test.ts',
      ],
      coverageDirectory: 'coverage/project-default',
      coverageReporters: ['lcov', 'text', 'html', 'json', 'json-summary'],
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
      displayName: 'rag-index-scripts',
      testMatch: ['**/rag-index/**/*.test.ts'],
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
        'scripts/agent-customization/mcp/neataptic-gate-mcp.mjs',
        'scripts/agent-customization/gates/code-coverage.gate.mjs',
        'scripts/agent-customization/gates/merge-coverage-summaries.mjs',
      ],
      coverageDirectory: 'coverage/project-agent-customization-scripts',
      coverageReporters: ['lcov', 'text', 'html', 'json', 'json-summary'],
    },
    {
      displayName: 'agent-customization-mjs',
      testMatch: ['**/scripts/agent-customization/**/*.test.mjs'],
      testEnvironment: 'node',
      transform: {},
      coverageProvider: 'v8',
      coverageDirectory: 'coverage/project-agent-customization-mjs',
      coverageReporters: ['lcov', 'text', 'html', 'json', 'json-summary'],
      collectCoverageFrom: [
        'scripts/agent-customization/gates/code-coverage.gate.mjs',
        'scripts/agent-customization/gates/merge-coverage-summaries.mjs',
        'scripts/agent-customization/mcp/neataptic-gate-mcp.mjs',
      ],
      coveragePathIgnorePatterns: ['<rootDir>/src/'],
      testTimeout: 300000,
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
      collectCoverageFrom: ['scripts/mcp-semantic/tools/submit-feedback.mjs'],
      coverageDirectory: 'coverage/project-mcp-semantic-scripts',
      coverageReporters: ['lcov', 'text', 'html', 'json', 'json-summary'],
    },
    {
      displayName: 'rag-index-mjs',
      testMatch: ['**/rag-index/**/*.test.mjs'],
      testEnvironment: 'node',
      transform: {},
      testTimeout: 600000,
    },
    {
      displayName: 'mcp-semantic-mjs',
      testMatch: ['**/scripts/mcp-semantic/**/*.test.mjs'],
      testEnvironment: 'node',
      transform: {},
      testTimeout: 300000,
      setupFilesAfterEnv: ['<rootDir>/testing/jest-setup-mcp-semantic.mjs'],
      collectCoverageFrom: ['scripts/mcp-semantic/tools/submit-feedback.mjs'],
      coverageDirectory: 'coverage/project-mcp-semantic-mjs',
      coverageReporters: ['lcov', 'text', 'html', 'json', 'json-summary'],
      coverageProvider: 'v8',
      coveragePathIgnorePatterns: ['<rootDir>/src/'],
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
    'scripts/agent-customization/mcp/neataptic-gate-mcp.mjs',
    'scripts/agent-customization/gates/code-coverage.gate.mjs',
    'scripts/agent-customization/gates/merge-coverage-summaries.mjs',
    'scripts/mcp-semantic/tools/submit-feedback.mjs',
  ],
  coverageReporters: ['lcov', 'text', 'html', 'json-summary'],
  testPathIgnorePatterns: ['/node_modules/', '/dist/'],
  modulePathIgnorePatterns: ['/dist/'],
};

export default config;
