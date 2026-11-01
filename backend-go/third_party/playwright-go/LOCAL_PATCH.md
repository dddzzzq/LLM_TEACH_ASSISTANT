# Local Playwright Go patch

Source: https://github.com/mxschmitt/playwright-go/tree/775648cd1805

Module version: `v0.6201.2-0.20260914222347-775648cd1805` (Playwright driver 1.62.1).
The upstream LICENSE is preserved. Only runtime Go sources, module metadata and
`internal/safe` are included; upstream browser tests and examples are omitted.

The worker routes the first navigation of new tabs. Upstream request handlers can
run concurrently with `newPage`, reading `frame.page` before it is assigned.
`frame.page` now uses `atomic.Pointer[pageImpl]`, and `newPage` publishes the page
after initializing its fields and handlers. All reads use `Load`; frame attachment
uses `Store`. `frame_publication_test.go` covers concurrent request access.

No driver JavaScript, browser binaries, APIs or protocol behavior are changed.
Remove the module replacement once the upstream dependency includes this fix and
the worker's real-browser race tests pass against that version.
