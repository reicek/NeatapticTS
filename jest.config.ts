const config = {
  preset: 'ts-jest/presets/default-esm',
  testEnvironment: 'node',
  testMatch: ['**/test/**/*.ts'],
  moduleFileExtensions: ['ts', 'js', 'json'],
  extensionsToTreatAsEsm: ['.ts'],
  transform: {
    '^.+\\.ts$': [
      'ts-jest',
      {
        useESM: true,
        tsconfig: 'tsconfig.json',
      },
    ],
  },
  setupFilesAfterEnv: ["<rootDir>/test/utils/jest-setup.ts"],
  testTimeout: 30000,
  collectCoverageFrom: [
    "src/**/*.ts",
    "!src/**/*.d.ts"
  ],
  coverageReporters: ["lcov", "text", "html"],
  testPathIgnorePatterns: ["/node_modules/", "/dist/"],
  // Add verbose output option that can be enabled with JEST_VERBOSE=1
  verbose: process.env.JEST_VERBOSE === '1',
  // Allow selective verbose output via environment variable JEST_SHOW_CONSOLE_FOR
  globals: {
    __SHOW_CONSOLE_FOR__: process.env.JEST_SHOW_CONSOLE_FOR || ''
  }
};

module.exports = config;
