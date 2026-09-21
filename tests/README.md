# tests/

## Priority

1. **Live e2e** (`tests/e2e/`, `--e2e`, needs `DAYTONA_API_KEY`) — real SDK, real sandboxes.
2. **CPU contracts** — framework-neutral runtime / telemetry / Slime adapter.
3. **`FakeAsyncDaytona`** (`tests/fake_sdk.py`) — CI stand-in only; must match observed live SDK shapes.

See `AGENTS.md` → Testing philosophy.
