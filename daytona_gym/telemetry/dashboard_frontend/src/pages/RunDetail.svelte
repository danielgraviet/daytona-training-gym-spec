<script>
  import { onMount } from 'svelte';
  import LineChart from '../components/charts/LineChart.svelte';
  import BarChart from '../components/charts/BarChart.svelte';
  import { fetchCharts, fetchLive, fetchRun, navigate } from '../lib/api.js';

  let { stem } = $props();

  let live = $state(null);
  let detail = $state(null);
  let charts = $state(null);
  let error = $state(null);
  let clock = $state(Date.now());

  async function refresh() {
    try {
      live = await fetchLive(stem);
      if (live?.ready || live?.n_rollouts > 0) {
        detail = await fetchRun(stem);
        charts = await fetchCharts(stem);
      } else {
        detail = null;
        charts = null;
      }
      error = null;
    } catch (e) {
      error = String(e.message || e);
    }
  }

  onMount(() => {
    refresh();
    const poll = setInterval(refresh, 2000);
    const tick = setInterval(() => (clock = Date.now()), 1000);
    return () => {
      clearInterval(poll);
      clearInterval(tick);
    };
  });

  function statusClass(s) {
    if (s === 'completed' || s === 'ok') return 'ok';
    if (s === 'failed' || s === 'aborted') return 'bad';
    return 'warn';
  }

  let elapsed = $derived.by(() => {
    const terminal =
      live?.done ||
      live?.failed ||
      live?.status === 'completed' ||
      live?.status === 'failed' ||
      live?.phase === 'completed' ||
      live?.phase === 'failed';
    if (typeof live?.elapsed_s === 'number' && Number.isFinite(live.elapsed_s)) {
      return Math.max(0, Math.floor(live.elapsed_s));
    }
    // Terminal with no frozen duration — hide rather than tick forever.
    if (terminal) return null;
    const started = live?.started_at;
    if (!started) return null;
    return Math.max(0, Math.floor(clock / 1000 - Number(started)));
  });
</script>

<section>
  <p class="muted mb-2 text-sm">
    <button class="cursor-pointer border-0 bg-transparent p-0 text-[var(--accent)]" onclick={() => navigate('/')}>
      ← runs
    </button>
  </p>
  <header class="mb-4 flex flex-wrap items-end justify-between gap-3">
    <div>
      <p class="muted m-0 text-xs uppercase tracking-[0.2em]">Run</p>
      <h1 class="m-0 text-2xl font-semibold">{stem}</h1>
    </div>
    <div class="muted flex flex-wrap gap-4 text-sm">
      <span>
        status
        <strong class={statusClass(live?.status || live?.phase)}>
          {live?.status || live?.phase || '…'}
        </strong>
      </span>
      {#if elapsed != null}
        <span>elapsed <strong>{elapsed}s</strong></span>
      {/if}
      {#if live?.message}
        <span>{live.message}</span>
      {/if}
    </div>
  </header>

  {#if error}
    <p class="bad">{error}</p>
  {/if}

  {#if !live?.ready && !detail}
    <div class="panel mb-4 p-4">
      <h2 class="mt-0 text-base">Starting</h2>
      <ol class="muted m-0 list-decimal pl-5 text-sm">
        {#each live?.activity || [] as a}
          <li class="mb-1">
            <span class="text-[var(--text)]">{a.phase}</span> — {a.message}
          </li>
        {:else}
          <li>Waiting for progress…</li>
        {/each}
      </ol>
      {#if live?.log_tail?.length}
        <pre class="mt-3 max-h-48 overflow-auto rounded bg-black/40 p-3 text-xs text-[var(--muted)]">{live.log_tail.join('\n')}</pre>
      {/if}
    </div>
  {:else if charts}
    <div class="mb-4 grid gap-3 md:grid-cols-2">
      <LineChart title="Mean reward" data={charts.reward} yLabel="reward" />
      <LineChart title="Wall time / rollout" data={charts.wall} color="var(--warn)" yLabel="seconds" />
      <BarChart title="Status counts" data={charts.status_histogram} />
      <BarChart title="Wall decomposition (sum)" data={charts.wall_decomposition} />
    </div>
    {#if charts.gpu_utilization?.length}
      <div class="mb-4">
        <LineChart
          title="GPU utilization %"
          data={charts.gpu_utilization.map((g, i) => ({ x: i, y: g.y, rollout_id: `gpu${g.gpu ?? ''}` }))}
          color="var(--ok)"
        />
      </div>
    {/if}
  {/if}

  {#if detail?.rollouts?.length}
    <div class="panel overflow-hidden">
      <table class="w-full border-collapse text-left text-sm">
        <thead class="muted text-xs uppercase tracking-wide">
          <tr class="border-b border-[var(--border)]">
            <th class="px-3 py-2 font-medium">rollout</th>
            <th class="px-3 py-2 font-medium">status</th>
            <th class="px-3 py-2 font-medium">reward</th>
            <th class="px-3 py-2 font-medium">wall</th>
          </tr>
        </thead>
        <tbody>
          {#each detail.rollouts as r}
            <tr class="border-b border-[var(--border)]/60 hover:bg-white/5">
              <td class="px-3 py-2">
                <button
                  class="cursor-pointer border-0 bg-transparent p-0 text-[var(--accent)]"
                  onclick={() =>
                    navigate(`/run/${encodeURIComponent(stem)}/rollout/${encodeURIComponent(r.rollout_id)}`)}
                >
                  {r.rollout_id}
                </button>
              </td>
              <td class={`px-3 py-2 ${statusClass(r.status)}`}>{r.status}</td>
              <td class="px-3 py-2">{r.reward == null ? '—' : Number(r.reward).toFixed(3)}</td>
              <td class="px-3 py-2">{r.wall_seconds == null ? '—' : `${Number(r.wall_seconds).toFixed(1)}s`}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}
</section>
