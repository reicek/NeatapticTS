# Linting errors:

IMPORTANT:
- Fix them file by file, run a targeted lint to confirm and then immediately erase from this list.
- Fix 1 file, confirm that file doesn't trow lint errors, delete the errors from this file immediately, go to the next file.
- Only run targeted lint tests, to avoid large logs.
  Example `npx eslint src\neat\neat.speciation.ts` 

Full errors list:

```

D:\code-practice\NeatapticTS\test\examples\asciiMaze\asciiMaze.e2e.test.ts
   16:28  warning  Unexpected any. Specify a different type                    @typescript-eslint/no-explicit-any
   86:1   error    Use const or class constructors instead of named functions  prefer-arrow/prefer-arrow-functions
   86:10  error    'analizeWinner' is defined but never used                   @typescript-eslint/no-unused-vars
  153:3   error    Unexpected 'debugger' statement                             no-debugger
  175:31  warning  Unexpected any. Specify a different type                    @typescript-eslint/no-explicit-any
  199:3   error    Use const or class constructors instead of named functions  prefer-arrow/prefer-arrow-functions

D:\code-practice\NeatapticTS\test\examples\asciiMaze\asciiMaze.ts
  40:3  error  A `require()` style import is forbidden  @typescript-eslint/no-require-imports

D:\code-practice\NeatapticTS\test\examples\asciiMaze\browser-entry.ts
   42:1   error    Use const or class constructors instead of named functions  prefer-arrow/prefer-arrow-functions
  123:8   error    Use const or class constructors instead of named functions  prefer-arrow/prefer-arrow-functions
  150:19  warning  Unexpected any. Specify a different type                    @typescript-eslint/no-explicit-any
  151:22  warning  Unexpected any. Specify a different type                    @typescript-eslint/no-explicit-any
  156:17  warning  Unexpected any. Specify a different type                    @typescript-eslint/no-explicit-any
  172:29  warning  Unexpected any. Specify a different type                    @typescript-eslint/no-explicit-any
  187:27  warning  Unexpected any. Specify a different type                    @typescript-eslint/no-explicit-any
  236:33  warning  Unexpected any. Specify a different type                    @typescript-eslint/no-explicit-any
  241:35  warning  Unexpected any. Specify a different type                    @typescript-eslint/no-explicit-any
  244:34  warning  Unexpected any. Specify a different type                    @typescript-eslint/no-explicit-any
  276:30  warning  Unexpected any. Specify a different type                    @typescript-eslint/no-explicit-any
  292:36  warning  Unexpected any. Specify a different type                    @typescript-eslint/no-explicit-any
  410:35  warning  Unexpected any. Specify a different type                    @typescript-eslint/no-explicit-any
  413:36  warning  Unexpected any. Specify a different type                    @typescript-eslint/no-explicit-any
  416:24  warning  Unexpected any. Specify a different type                    @typescript-eslint/no-explicit-any
  418:45  warning  Unexpected any. Specify a different type                    @typescript-eslint/no-explicit-any
  474:45  warning  Unexpected any. Specify a different type                    @typescript-eslint/no-explicit-any
  475:39  warning  Unexpected any. Specify a different type                    @typescript-eslint/no-explicit-any
  484:15  error    '__webpack_require__' is defined but never used             @typescript-eslint/no-unused-vars
  484:36  warning  Unexpected any. Specify a different type                    @typescript-eslint/no-explicit-any
  485:49  warning  Unexpected any. Specify a different type                    @typescript-eslint/no-explicit-any
  486:23  warning  Unexpected any. Specify a different type                    @typescript-eslint/no-explicit-any
  486:39  warning  Unexpected any. Specify a different type                    @typescript-eslint/no-explicit-any
  490:55  warning  Unexpected any. Specify a different type                    @typescript-eslint/no-explicit-any

D:\code-practice\NeatapticTS\test\examples\asciiMaze\browserLogger.ts
  107:1   error    Use const or class constructors instead of named functions   prefer-arrow/prefer-arrow-functions
  122:1   error    Use const or class constructors instead of named functions   prefer-arrow/prefer-arrow-functions
  154:32  error    Unexpected control character(s) in regular expression: \x1b  no-control-regex
  434:8   error    Use const or class constructors instead of named functions   prefer-arrow/prefer-arrow-functions
  436:14  warning  Unexpected any. Specify a different type                     @typescript-eslint/no-explicit-any
  454:20  warning  Unexpected any. Specify a different type                     @typescript-eslint/no-explicit-any
  464:21  warning  Unexpected any. Specify a different type                     @typescript-eslint/no-explicit-any
  466:55  warning  Unexpected any. Specify a different type                     @typescript-eslint/no-explicit-any
  470:39  warning  Unexpected any. Specify a different type                     @typescript-eslint/no-explicit-any
  472:38  warning  Unexpected any. Specify a different type                     @typescript-eslint/no-explicit-any

D:\code-practice\NeatapticTS\test\examples\asciiMaze\evolutionEngine.ts
   63:22  warning  Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
  154:39  warning  Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
  178:17  warning  Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
  179:15  warning  Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
  181:6   warning  Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
  398:27  warning  Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
  437:34  warning  Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any

D:\code-practice\NeatapticTS\test\examples\asciiMaze\evolutionEngine\sampling.ts
  179:3   warning  Unused eslint-disable directive (no problems were reported from '@typescript-eslint/no-explicit-any')
  181:30  warning  Unexpected any. Specify a different type                                                               @typescript-eslint/no-explicit-any
  182:35  warning  Unexpected any. Specify a different type                                                               @typescript-eslint/no-explicit-any

D:\code-practice\NeatapticTS\test\examples\asciiMaze\evolutionEngine\telemetryMetrics.ts
  151:12  warning  Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
  200:9   warning  Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
  202:12  warning  Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
  211:11  warning  Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
  353:9   warning  Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
  405:9   warning  Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
  430:21  warning  Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
  626:9   warning  Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
  630:21  warning  Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
  739:10  warning  Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
  771:10  warning  Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any

D:\code-practice\NeatapticTS\test\examples\asciiMaze\interfaces.ts
    52:20  warning  Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
   915:18  warning  Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
  1043:20  warning  Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
```