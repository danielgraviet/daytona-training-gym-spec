<script>
  import { navigate } from '../lib/api.js';

  let { analysis, stem } = $props();

  const width = 960;
  const pad = { l: 44, r: 12 };
  const gpuH = 56;
  const rowH = 9;
  const rowGap = 3;
  const maxRows = 80;

  let totals = $derived(analysis?.totals || {});
  let cost = $derived(analysis?.cost);
  let defs = $derived(analysis?.definitions || {});
  let steps = $derived(analysis?.steps || []);
  // Steps don't overlap in time, so each step reuses the same row slots.
  let rows = $derived.by(() => {
    const seen = {};
    return (analysis?.rollouts || [])
      .map((r) => {
        const lane = seen[r.step] ?? 0;
        seen[r.step] = lane + 1;
        return { ...r, lane };
      })
      .filter((r) => r.lane < maxRows);
  });
  let laneCount = $derived(rows.reduce((m, r) => Math.max(m, r.lane + 1), 0));

  let domain = $derived.by(() => {
    if (!rows.length) return null;
    const t0 = Math.min(...rows.map((r) => r.start));
    const t1 = Math.max(...rows.map((r) => r.end));
    return { t0, t1, span: Math.max(1e-6, t1 - t0) };
  });

  function x(t) {
    return pad.l + ((t - domain.t0) / domain.span) * (width - pad.l - pad.r);
  }

  let rowsTop = gpuH + 14;
  let height = $derived(rowsTop + laneCount * (rowH + rowGap) + 22);

  let gpuPath = $derived.by(() => {
    if (!domain) return '';
    const pts = (analysis?.gpu_utilization || []).filter(
      (g) => g.t >= domain.t0 - 1 && g.t <= domain.t1 + 1,
    );
    if (pts.length < 2) return '';
    return pts
      .map((g, i) => {
        const px = Math.min(width - pad.r, Math.max(pad.l, x(g.t)));
        const py = 4 + (1 - Math.min(100, Math.max(0, g.y)) / 100) * (gpuH - 8);
        return `${i === 0 ? 'M' : 'L'}${px.toFixed(1)},${py.toFixed(1)}`;
      })
      .join(' ');
  });

  let ticks = $derived.by(() => {
    if (!domain) return [];
    return [0, 0.25, 0.5, 0.75, 1].map((f) => ({
      px: pad.l + f * (width - pad.l - pad.r),
      label: fmtSecs(f * domain.span),
    }));
  });

  function fmtSecs(s) {
    if (s == null || !Number.isFinite(s)) return '—';
    if (s >= 3600) return `${(s / 3600).toFixed(1)}h`;
    if (s >= 60) return `${(s / 60).toFixed(1)}m`;
    return `${s.toFixed(s < 10 ? 2 : 1)}s`;
  }
  function fmtUsd(v) {
    if (v == null) return '—';
    return v < 1 ? `$${v.toFixed(4)}` : `$${v.toFixed(2)}`;
  }
  function pct(v) {
    return `${Math.round((v || 0) * 100)}%`;
  }
  let idleFrac = $derived(
    totals.rollout_phase_seconds > 0 ? totals.env_wait_seconds / totals.rollout_phase_seconds : 0,
  );
</script>

{#if analysis && totals.n_rollouts}
  <section class="mb-4">
    <h2 class="mb-2 mt-0 text-base font-semibold">Where the time went</h2>
    <div class="mb-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
      <div class="panel p-3" title={defs.env_wait_seconds}>
        <p class="muted m-0 text-xs uppercase tracking-wide">GPU idle waiting on envs</p>
        <p class="m-0 text-2xl font-semibold {idleFrac >= 0.5 ? 'bad' : 'ok'}">{pct(idleFrac)}</p>
        <p class="muted m-0 text-xs">{fmtSecs(totals.env_wait_seconds)} of {fmtSecs(totals.rollout_phase_seconds)} rollout phase</p>
      </div>
      <div class="panel p-3" title={defs.idle_gpu_cost}>
        <p class="muted m-0 text-xs uppercase tracking-wide">Idle GPU cost</p>
        {#if cost}
          <p class="m-0 text-2xl font-semibold">{fmtUsd(cost.idle_gpu_cost)}</p>
          <p class="muted m-0 text-xs">
            of {fmtUsd(cost.run_cost)} · {cost.num_gpus}× GPU @ {fmtUsd(cost.gpu_cost_per_hour)}/h
          </p>
        {:else}
          <p class="muted m-0 text-sm">Set <code>gpu_cost_per_hour</code> on TrainConfig</p>
        {/if}
      </div>
      <div class="panel p-3" title={defs.straggler_tax_seconds}>
        <p class="muted m-0 text-xs uppercase tracking-wide">Straggler tax</p>
        <p class="m-0 text-2xl font-semibold">{fmtSecs(totals.straggler_tax_seconds)}</p>
        <p class="muted m-0 text-xs">across {totals.n_steps} step(s)</p>
      </div>
      <div class="panel p-3">
        <p class="muted m-0 text-xs uppercase tracking-wide">Sandbox provision p95</p>
        <p class="m-0 text-2xl font-semibold">{fmtSecs(totals.sandbox_provision?.p95)}</p>
        <p class="muted m-0 text-xs">p50 {fmtSecs(totals.sandbox_provision?.p50)} · n={totals.sandbox_provision?.n}</p>
      </div>
      <div class="panel p-3" title={defs.bound}>
        <p class="muted m-0 text-xs uppercase tracking-wide">Bottleneck</p>
        <p class="m-0 text-2xl font-semibold {totals.bound === 'environment' ? 'warn' : ''}">{totals.bound}</p>
        {#if totals.n_failed}
          <p class="muted m-0 text-xs" title={defs.failed_waste_seconds}>
            {fmtSecs(totals.failed_waste_seconds)} wasted on {totals.n_failed} failed
          </p>
        {/if}
      </div>
    </div>

    {#if domain}
      <div class="panel mb-3 p-3">
        <div class="mb-1 flex flex-wrap items-baseline justify-between gap-2">
          <h3 class="m-0 text-sm font-semibold">Rollout timeline</h3>
          <span class="muted text-xs">
            <span class="legend" style="background: var(--accent)"></span> inference
            <span class="legend ml-3" style="background: var(--border)"></span> sandbox / tools / reward
            <span class="legend ml-3" style="background: var(--bad); opacity: .35"></span> GPU idle on envs
            <span class="legend ml-3" style="background: var(--ok)"></span> GPU util
          </span>
        </div>
        <svg viewBox={`0 0 ${width} ${height}`} class="h-auto w-full" role="img" aria-label="rollout timeline">
          <text x={pad.l - 6} y={12} text-anchor="end" fill="var(--muted)" font-size="10">100%</text>
          <text x={pad.l - 6} y={gpuH} text-anchor="end" fill="var(--muted)" font-size="10">0%</text>
          <line x1={pad.l} x2={width - pad.r} y1={gpuH} y2={gpuH} stroke="var(--border)" />
          {#if gpuPath}
            <path d={gpuPath} fill="none" stroke="var(--ok)" stroke-width="1.5" />
          {:else}
            <text x={pad.l + 6} y={gpuH / 2 + 4} fill="var(--muted)" font-size="11">no GPU samples in this window</text>
          {/if}

          {#each steps as s}
            {#each s.env_wait_windows || [] as w}
              <rect
                x={x(w[0])}
                y={rowsTop - 4}
                width={Math.max(0.5, x(w[1]) - x(w[0]))}
                height={laneCount * (rowH + rowGap) + 4}
                fill="var(--bad)"
                opacity="0.14"
              />
            {/each}
          {/each}

          {#each rows as r}
            {@const y = rowsTop + r.lane * (rowH + rowGap)}
            <g
              class="cursor-pointer"
              role="link"
              tabindex="0"
              onclick={() => navigate(`/run/${encodeURIComponent(stem)}/rollout/${encodeURIComponent(r.rollout_id)}`)}
              onkeydown={(e) => e.key === 'Enter' && navigate(`/run/${encodeURIComponent(stem)}/rollout/${encodeURIComponent(r.rollout_id)}`)}
            >
              <title>{r.rollout_id} · step {r.step} · {r.status} · {fmtSecs(r.wall)}</title>
              <rect
                x={x(r.start)}
                {y}
                width={Math.max(1, x(r.end) - x(r.start))}
                height={rowH}
                rx="2"
                fill="var(--border)"
                stroke={['failed', 'aborted', 'cancelled'].includes(r.status) ? 'var(--bad)' : 'none'}
              />
              {#each r.inference_intervals as iv}
                <rect x={x(iv[0])} {y} width={Math.max(0.8, x(iv[1]) - x(iv[0]))} height={rowH} fill="var(--accent)" />
              {/each}
            </g>
          {/each}

          {#each ticks as t}
            <text x={t.px} y={height - 6} text-anchor="middle" fill="var(--muted)" font-size="10">{t.label}</text>
          {/each}
        </svg>
        {#if rows.length < (analysis.rollouts || []).length}
          <p class="muted m-0 text-xs">showing first {maxRows} rollouts per step</p>
        {/if}
      </div>
    {/if}

    {#if steps.length > 1}
      <div class="panel mb-3 overflow-x-auto">
        <table class="w-full border-collapse text-left text-sm">
          <thead class="muted text-xs uppercase tracking-wide">
            <tr class="border-b border-[var(--border)]">
              <th class="px-3 py-2 font-medium">step</th>
              <th class="px-3 py-2 font-medium">rollouts</th>
              <th class="px-3 py-2 font-medium">rollout phase</th>
              <th class="px-3 py-2 font-medium" title={defs.env_wait_seconds}>GPU idle on envs</th>
              <th class="px-3 py-2 font-medium" title={defs.straggler_tax_seconds}>straggler</th>
              <th class="px-3 py-2 font-medium" title={defs.train_phase_seconds}>train (derived)</th>
              <th class="px-3 py-2 font-medium">slowest rollout</th>
              <th class="px-3 py-2 font-medium">bound</th>
            </tr>
          </thead>
          <tbody>
            {#each steps as s}
              <tr class="border-b border-[var(--border)]/60">
                <td class="px-3 py-2">{s.step}</td>
                <td class="px-3 py-2">{s.n_rollouts}{s.n_failed ? ` (${s.n_failed} failed)` : ''}</td>
                <td class="px-3 py-2">{fmtSecs(s.rollout_phase_seconds)}</td>
                <td class="px-3 py-2">{fmtSecs(s.env_wait_seconds)} <span class="muted">({pct(s.env_wait_fraction)})</span></td>
                <td class="px-3 py-2">{fmtSecs(s.straggler_tax_seconds)}</td>
                <td class="px-3 py-2">{fmtSecs(s.train_phase_seconds)}</td>
                <td class="px-3 py-2">
                  <button
                    class="cursor-pointer border-0 bg-transparent p-0 text-[var(--accent)]"
                    onclick={() => navigate(`/run/${encodeURIComponent(stem)}/rollout/${encodeURIComponent(s.critical_rollout_id)}`)}
                  >{s.critical_rollout_id}</button>
                </td>
                <td class="px-3 py-2 {s.bound === 'environment' ? 'warn' : ''}">{s.bound}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}

    <details class="muted text-xs">
      <summary class="cursor-pointer">How these are computed</summary>
      <dl class="mt-2">
        {#each Object.entries(defs) as [k, v]}
          <dt class="text-[var(--text)]">{k}</dt>
          <dd class="mb-2 ml-4">{v}</dd>
        {/each}
      </dl>
    </details>
  </section>
{/if}

<style>
  .legend {
    display: inline-block;
    width: 10px;
    height: 8px;
    border-radius: 2px;
    vertical-align: middle;
  }
</style>
