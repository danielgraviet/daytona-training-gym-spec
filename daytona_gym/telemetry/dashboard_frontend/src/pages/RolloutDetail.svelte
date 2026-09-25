<script>
  import { onMount } from 'svelte';
  import BarChart from '../components/charts/BarChart.svelte';
  import { fetchRollout, navigate } from '../lib/api.js';

  let { stem, rolloutId } = $props();
  let detail = $state(null);
  let error = $state(null);

  onMount(async () => {
    try {
      detail = await fetchRollout(stem, rolloutId);
    } catch (e) {
      error = String(e.message || e);  // OfflineError already has a readable message
    }
  });
</script>

<section>
  <p class="muted mb-2 text-sm">
    <button
      class="cursor-pointer border-0 bg-transparent p-0 text-[var(--accent)]"
      onclick={() => navigate(`/run/${encodeURIComponent(stem)}`)}
    >
      ← {stem}
    </button>
  </p>
  <header class="mb-4">
    <p class="muted m-0 text-xs uppercase tracking-[0.2em]">Rollout</p>
    <h1 class="m-0 text-2xl font-semibold">{rolloutId}</h1>
  </header>

  {#if error}
    <p class="bad">{error}</p>
  {:else if !detail}
    <p class="muted">Loading…</p>
  {:else}
    <div class="muted mb-4 flex flex-wrap gap-4 text-sm">
      <span>status <strong class={detail.status === 'completed' ? 'ok' : 'bad'}>{detail.status}</strong></span>
      <span>reward <strong>{detail.reward == null ? '—' : Number(detail.reward).toFixed(3)}</strong></span>
      <span>wall <strong>{detail.wall_seconds == null ? '—' : `${Number(detail.wall_seconds).toFixed(2)}s`}</strong></span>
    </div>

    <div class="mb-4">
      <BarChart title="Wall decomposition" data={detail.wall_decomposition || {}} />
    </div>

    <div class="panel overflow-hidden">
      <table class="w-full border-collapse text-left text-sm">
        <thead class="muted text-xs uppercase tracking-wide">
          <tr class="border-b border-[var(--border)]">
            <th class="px-3 py-2 font-medium">step</th>
            <th class="px-3 py-2 font-medium">offset</th>
            <th class="px-3 py-2 font-medium">duration</th>
          </tr>
        </thead>
        <tbody>
          {#each detail.steps || [] as s}
            <tr class="border-b border-[var(--border)]/60">
              <td class="px-3 py-2">{s.short || s.name}</td>
              <td class="px-3 py-2">{Number(s.offset_seconds).toFixed(2)}s</td>
              <td class="px-3 py-2">{Number(s.duration_seconds).toFixed(2)}s</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}
</section>
