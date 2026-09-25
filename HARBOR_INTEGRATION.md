# Harbor Integration

Status: **gym backend stub** (`daytona_gym.gym.harbor.HarborBackend` under `TrainConfig(backend="harbor")`). Full Harbor×Daytona plugin not implemented yet — `launch()` raises a typed error. Not CLI-first.

Partner-facing north star is Modal-shaped (`TrainConfig.launch()`), **not** raw Harbor CLI. See `COMPETITIVE_MODAL_GYM.md` and `PRODUCT_DECISIONS.md` §2 / §15 / §16.

## Why Harbor (still)

Many teams already do RL / agent eval roughly like this:

```text
BYO GPU cluster (policy / inference)
        │
     Harbor  ──tasks (e.g. Terminal-Bench)──►  Daytona sandboxes
        │
   local metrics / logs (often incomplete across the boundary)
```

Harbor already knows how to talk to Daytona as a sandbox backend. Daytona Training Gym should not replace that harness. It should make the **cross-boundary critical path** obvious: GPU/inference time vs sandbox provision / tools / finalize, with stable correlation IDs — preferably behind the same gym API partners use for Slime.

That matches the product wedge:

> BYO GPUs + Harbor on Daytona + see where the minutes go.

## Adapter goals (when we build it)

1. Reuse shared telemetry (same span vocabulary as Slime dogfood); keep Harbor types out of core.
2. Plug into Harbor’s Daytona environment / sandbox configuration path — deepen, don’t fork Harbor core (`-e` import path and/or job plugin).
3. Emit: `sandbox.provision`, `sandbox.seed` (if used), `tool.*`, `sandbox.finalize`, plus inference correlation when available.
4. Prefer exposing Harbor through gym recipes / Python config. Do **not** make long `harbor run -e module:Class --plugin …` the documented happy path (escape hatch only).

## Non-goals (initially)

- Owning Terminal-Bench task definitions
- Replacing Harbor’s scheduler or reward logic
- Requiring Daytona-managed GPUs
- Shipping Harbor before the gym facade direction is clear
- A `dg harbor` CLI wrapper as the product surface

## Spike checklist (when chosen over Gym SDK skeleton)

- [ ] Map Harbor’s Daytona sandbox config surface (env vars / YAML / Python API)
- [ ] Identify the smallest hook to inject correlation IDs + Daytona Gym telemetry
- [ ] One CPU-only contract test with a fake Harbor-shaped caller
- [ ] One partner-readable timeline: Harbor trial ↔ Daytona spans in `dg` / inspect
- [ ] Wire under future `TrainConfig` / recipe (not CLI-first)

## Related

- `COMPETITIVE_MODAL_GYM.md` — why gym facade > Harbor CLI
- `PRODUCT_DECISIONS.md` — §2, §15, §16
- `ARCHITECTURE.md` — adapter diagram
- `SLIME_INTEGRATION.md` — pattern for “framework owns outer loop; Daytona owns env + traces”
- `examples/coding_dogfood/` — proven span timeline to mirror
