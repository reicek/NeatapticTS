# __rag_test__

A very large function that exceeds chunk size limits.
This function has many statements to trigger statement-group splitting.

## __rag_test__/index.ts

### helperFunc

```ts
helperFunc(
  x: number,
): number
```

Helper function.

Parameters:
- `x` - Input number.

Returns: Incremented value.

### HelperInterface

Helper interface.

## __rag_test__/large.ts

### smallFunc

```ts
smallFunc(): number
```

Small function.

### veryLargeFunction

```ts
veryLargeFunction(): string
```

A very large function that exceeds chunk size limits.
This function has many statements to trigger statement-group splitting.

## __rag_test__/module-a.ts

Helper function.

### helperArrow

```ts
helperArrow(
  x: string,
): string
```

Arrow function variable.

### helperConst

Non-function variable.

### helperDefault

```ts
helperDefault(): void
```

Default export.

### HelperError

Error class.

### helperFunc

```ts
helperFunc(
  x: number,
): number
```

Helper function.

Parameters:
- `x` - Input number.

Returns: Incremented value.

### HelperInterface

Helper interface.

### HelperType

Type alias.

### module-a()

```ts
module-a()(): void
```

Default export.

## __rag_test__/module-b.ts

### Consumer

Consumer class implementing HelperInterface.

#### getValue

```ts
getValue(): string
```

Get value.

#### useHelper

```ts
useHelper(): number
```

Use helper function with additional computation.
This method body is intentionally long enough to exceed the minimum
method entity character threshold so that a separate function entity
(with parent_class metadata) is produced for it during code entity
extraction.

### standaloneFunc

```ts
standaloneFunc(): number
```

Standalone function calling helper.
