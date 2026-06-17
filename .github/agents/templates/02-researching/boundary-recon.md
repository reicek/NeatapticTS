---
title: Boundary Recon Packet
description: Use to ask scouts for a focused boundary map of a source file or folder.
fields:
  - name: plan_ref
    type: string
    description: Path to the active plan step (e.g. plans/feature-x.step01.md)
  - name: file_paths
    type: array
    items: string
    description: One or more repo paths to inspect
  - name: question
    type: string
    description: The focused research question
timeout_seconds: 30
concurrency: 1
expected_output: structured-v1
---

Example:

```json
{
  "plan_ref": "plans/feature-x.step01.md",
  "file_paths": ["src/feature/x/controller.ts"],
  "question": "List public methods used by external callers and dependencies."
}
```
