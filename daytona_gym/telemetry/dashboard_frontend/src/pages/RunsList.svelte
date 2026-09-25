<script>
  import { onMount } from 'svelte';
  import { fetchRuns, navigate } from '../lib/api.js';

  let runs = $state([]);
  let error = $state(null);
  let clock = $state(Date.now());

  async function refresh() {
    try {
      runs = await fetchRuns();
      error = null;
    } catch (e) {
      error = String(e.message || e);
    }
  }

  onMount(() => {
    refresh();
    const poll = setInterval(refresh, 2500);
    const tick = setInterval(() => (clock = Date.now()), 1000);
    return () => {
      clearInterval(poll);
      clearInterval(tick);
    };
  });

  function statusClass(s) {
    if (s === 'completed') return 'ok';
    if (s === 'failed') return 'bad';
    if (s === 'running') return 'warn';
    return 'muted';
  }

  function age(ts) {
    if (!ts) return '—';
    const sec = Math.max(0, Math.floor((clock / 1000) - Number(ts)));
    if (sec < 60) return `${sec}s ago`;
    if (sec < 3600) return `${Math.floor(sec / 60)}m ago`;
    return `${Math.floor(sec / 3600)}h ago`;
  }
</script>

<section>
  <header class="mb-4 flex items-end justify-between gap-4">
    <div>
      <p class="muted m-0 text-xs uppercase tracking-[0.2em]">Daytona Gym</p>
      <h1 class="m-0 text-2xl font-semibold">Training runs</h1>
    </div>
    <p class="muted m-0 text-xs">live · poll 2.5s</p>
  </header>

  {#if error}
    <p class="bad">{error}</p>
  {/if}

  <div class="panel overflow-hidden">
    <table class="w-full border-collapse text-left text-sm">
      <thead class="muted text-xs uppercase tracking-wide">
        <tr class="border-b border-[var(--border)]">
          <th class="px-3 py-2 font-medium">run</th>
          <th class="px-3 py-2 font-medium">status</th>
          <th class="px-3 py-2 font-medium">n</th>
          <th class="px-3 py-2 font-medium">mean reward</th>
          <th class="px-3 py-2 font-medium">updated</th>
        </tr>
      </thead>
      <tbody>
        {#each runs as run}
          <tr class="border-b border-[var(--border)]/60 hover:bg-white/5">
            <td class="px-3 py-2">
              <button
                class="cursor-pointer border-0 bg-transparent p-0 text-[var(--accent)]"
                onclick={() => navigate(`/run/${encodeURIComponent(run.stem)}`)}
              >
                {run.stem}
              </button>
            </td>
            <td class={`px-3 py-2 ${statusClass(run.run_status)}`}>
              {run.run_status || '—'}
              {#if run.phase && run.run_status === 'running'}
                <span class="muted"> · {run.phase}</span>
              {/if}
            </td>
            <td class="px-3 py-2">{run.n_rollouts ?? 0}</td>
            <td class="px-3 py-2">
              {run.mean_reward == null ? '—' : Number(run.mean_reward).toFixed(3)}
            </td>
            <td class="muted px-3 py-2">{age(run.progress_updated_at || run.mtime)}</td>
          </tr>
        {:else}
          <tr>
            <td class="muted px-3 py-6" colspan="5">No runs yet — launch a job to populate runs/</td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
</section>
