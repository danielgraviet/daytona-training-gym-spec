# Harbor Integration

Status: **planned second adapter** (after Slime MVP). Spec note only — not implemented yet.

## Why Harbor next

Many teams already do RL / agent eval roughly like this:

```text
BYO GPU cluster (policy / inference)
        │
     Harbor fork  ──tasks (e.g. Terminal-Bench)──►  Daytona sandboxes
        │
   local metrics / logs (often incomplete across the boundary)
```

Harbor already knows how to talk to Daytona as a sandbox backend. Daytona Training Gym should not replace that harness. It should make the **cross-boundary critical path** obvious: GPU/inference time vs sandbox provision / tools / finalize, with stable correlation IDs.

That matches the product wedge in `PRODUCT_DECISIONS.md`:

> BYO GPUs + Harbor on Daytona + see where the minutes go.

## Adapter goals (when we build it)

1. Reuse the framework-neutral `EnvironmentRuntime` + telemetry (same as Slime).
2. Plug into Harbor’s Daytona environment / sandbox configuration path — deepen, don’t fork Harbor core.
3. Emit the same span vocabulary buyers already saw in Slime dogfood: `sandbox.provision`, `sandbox.seed` (if used), `tool.*`, `sandbox.finalize`, plus inference correlation when available.
4. Keep Harbor/Terminal-Bench types out of Daytona core; convert at `daytona_gym/adapters/harbor/`.

## Non-goals (initially)

- Owning Terminal-Bench task definitions
- Replacing Harbor’s scheduler or reward logic
- Requiring Daytona-managed GPUs
- Shipping Harbor before Slime MVP polish is “good enough to show”

## Spike checklist (next engineering session)

- [ ] Map Harbor’s Daytona sandbox config surface (env vars / YAML / Python API)
- [ ] Identify the smallest hook to inject correlation IDs + Daytona Gym telemetry
- [ ] One CPU-only contract test with a fake Harbor-shaped caller
- [ ] One partner-readable timeline: Harbor rollout ↔ Daytona spans in `inspect`

## Related

- `PRODUCT_DECISIONS.md` — §2 (Slime first, Harbor second), §15 (deepen Harbor)
- `ARCHITECTURE.md` — adapter diagram
- `SLIME_INTEGRATION.md` — pattern for “framework owns outer loop; Daytona owns env + traces”
- `examples/coding_dogfood/` — proven span timeline to mirror
