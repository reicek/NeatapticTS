---
title: Prior Art Scan Packet
description: Use to collect prior art, academic references, and existing implementations related to a feature.
fields:
  - name: keywords
    type: array
    items: string
    description: Search keywords and phrases
  - name: max_results
    type: integer
    description: Maximum number of external results to fetch
timeout_seconds: 60
concurrency: 1
expected_output: structured-v1
---

Example:

```json
{
  "keywords": ["NEAT algorithm", "speciation NEAT 2002"],
  "max_results": 10
}
```
