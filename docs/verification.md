# Delivery verification — 2026-10-29

- Server suite: **263 passed**, 14 live-integration tests deselected.
- Existing ingestion, storage, embedding reuse, retrieval and conversation tests
  are included; new tests cover HTTP validation, JSON/NDJSON contracts, UTF-8,
  API-key protection, readiness, schema exposure and bootstrap write safeguards.
- Local snapshot: **22,988 vectors**, dimension **1,024**, **10,550 document IDs**.
  Mapping and BM25 chunk identity sets match.
- Current and archived retrieval inventories were validated; recorded archive
  file hashes were checked against copied artifacts.
- All eight local server packages were installed in a separate verification
  environment without downloading dependencies. Existing dependency libraries
  were reused; local module paths resolved within this delivery.
- No UI package/assets are included. Local Markdown links were checked.

Verification did not start live model inference, rebuild the development Qdrant
collection, run collection, or execute a real data update. Target-server package
downloads, model availability, live end-to-end checks and load testing remain
deployment acceptance steps. See [deployment](deployment.md).
