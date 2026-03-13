# External Sources And Wikimedia Media Guide

Use this guide when enriching documentation with external reading, citations, or
images.

## Principle

External material should deepen understanding without weakening trust,
maintainability, or licensing compliance.

## Source Selection Rules

Prefer sources in this order:

1. The repo's own source code and generated docs.
2. Standards, specifications, or canonical project documentation.
3. High-value background reading, including Wikipedia, when it helps a reader
   understand a concept that the repo relies on.

Use external sources when they teach a concept. Do not use them to explain the
repo's own behavior when the code is the better authority.

## Citation Requirements

Whenever you rely on an external source, record:

- title,
- author or responsible organization,
- canonical URL,
- why the source is being cited,
- license details when the reused material requires attribution.

### Compact citation pattern for prose

Use readable prose first, then a compact attribution note.

Example:

```text
For background on graph theory, see Wikipedia contributors, "Graph theory,"
Wikipedia, The Free Encyclopedia.
```

### Wikipedia article guidance

- Attribute article text or background references to `Wikipedia contributors`.
- Prefer linking the article title rather than dropping raw URLs into the prose.
- Treat the article as conceptual background, not as proof of repo behavior.
- If you quote or closely paraphrase article text, make the attribution
  explicit and keep the excerpt short.

## Wikimedia Commons Media Workflow

Prefer files hosted on Wikimedia Commons, because the file description page is
where license and creator data live.

### Required checks before use

1. Open the exact file page.
2. Verify the file is hosted on Wikimedia Commons, not only on a local
   Wikipedia project.
3. Verify the file is under a compatible free license or is public domain.
4. Confirm whether the creator is the original author or whether the uploader is
   merely a re-publisher.
5. Check whether the license requires attribution, ShareAlike, or a linked copy
   of the license text.
6. Check for non-copyright restrictions such as personality rights, trademarks,
   or moral-rights concerns.
7. If the file is a derivative work, confirm that the derivative is allowed.
8. If any part is unclear, do not use the media.

### Reject immediately if any of these are true

- The file is fair use or non-free.
- The file is `CC BY-NC`, `CC BY-ND`, `CC BY-NC-SA`, or `CC BY-NC-ND`.
- The file page does not clearly identify the author or license.
- The educational value is minor and the compliance burden is high.

## Attribution Bundle For Media

If a file is used, keep an attribution note with all of the following:

- file title,
- original creator,
- source page URL,
- license name,
- license URL when required,
- whether the image was modified,
- whether ShareAlike obligations apply.

### Attribution template

```text
Image: "<file title>" by <creator>, via Wikimedia Commons.
License: <license name> (<license URL>).
Source: <file page URL>.
Modified: <yes or no>.
```

## Why These Rules Exist

These rules are informed by Wikimedia's own reuse guidance:

- Wikimedia Commons contributors, "Commons:Reusing content outside Wikimedia,"
  Wikimedia Commons, https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia, accessed 2026-03-12.
- Wikimedia Commons contributors, "Commons:Licensing," Wikimedia Commons,
  https://commons.wikimedia.org/wiki/Commons:Licensing, accessed 2026-03-12.
- Wikipedia contributors, "Technical writing," Wikipedia, The Free
  Encyclopedia, https://en.wikipedia.org/wiki/Technical_writing, accessed
  2026-03-12.

Key takeaways reflected in this skill:

- credit the original creator, not automatically the uploader,
- identify the exact license and follow attribution requirements,
- respect ShareAlike obligations for derivatives,
- remember that non-copyright restrictions may still matter,
- use images only when they genuinely improve understanding.
