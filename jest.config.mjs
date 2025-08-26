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
      testMatch: ['**/test/**/!(asciiMaze).*.(test).ts'],
      preset: 'ts-jest/presets/default-esm',
      testEnvironment: 'node',
      extensionsToTreatAsEsm: ['.ts'],
      transform: {
        '^.+\\.ts$': [
          'ts-jest',
          { useESM: true, tsconfig: 'tsconfig.test.json', diagnostics: true }
        ]
      },
      setupFilesAfterEnv: ['<rootDir>/test/utils/jest-setup.ts'],
      testTimeout: 300000,
      collectCoverageFrom: ['src/**/*.ts', '!src/**/*.d.ts'],
      coverageReporters: ['lcov', 'text', 'html'],
      testPathIgnorePatterns: ['/node_modules/', '/dist/'],
      verbose: process.env.JEST_VERBOSE === '1'
    },
    {
      displayName: 'asciiMaze-browser',
      testMatch: ['**/test/examples/asciiMaze/**/*.test.ts'],
      preset: 'ts-jest/presets/default-esm',
      testEnvironment: 'jsdom',
      extensionsToTreatAsEsm: ['.ts'],
      transform: {
        '^.+\\.ts$': [
          'ts-jest',
          { useESM: true, tsconfig: 'tsconfig.test.json', diagnostics: true }
        ]
      },
      setupFilesAfterEnv: ['<rootDir>/test/utils/jest-setup.ts'],
      testTimeout: 3000000
    }
  ],
  testMatch: ['**/test/**/*.test.ts'],
  moduleFileExtensions: ['ts', 'js', 'mjs', 'cjs', 'json'],
  extensionsToTreatAsEsm: ['.ts'],
  transform: {
    '^.+\\.ts$': [
      'ts-jest',
      {
        useESM: true,
        tsconfig: 'tsconfig.test.json',
        diagnostics: true
      }
    ]
  },
  setupFilesAfterEnv: ['<rootDir>/test/utils/jest-setup.ts'],
  testTimeout: 300000,
  collectCoverageFrom: ['src/**/*.ts', '!src/**/*.d.ts'],
  coverageReporters: ['lcov', 'text', 'html'],
  testPathIgnorePatterns: ['/node_modules/', '/dist/'],
  verbose: process.env.JEST_VERBOSE === '1',
  globals: {
    __SHOW_CONSOLE_FOR__: process.env.JEST_SHOW_CONSOLE_FOR || ''
  }
};

export default config;
