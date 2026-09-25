<script>
  let { title = '', data = {}, height = 140 } = $props();

  const colors = ['var(--accent)', 'var(--ok)', 'var(--warn)', 'var(--bad)', '#9b7bff', '#5ad4e6'];

  let entries = $derived(
    Object.entries(data || {})
      .map(([k, v]) => [k, Number(v) || 0])
      .filter(([, v]) => v > 0)
      .sort((a, b) => b[1] - a[1]),
  );

  let total = $derived(entries.reduce((s, [, v]) => s + v, 0) || 1);
  const width = 640;
  const pad = { t: 12, r: 12, b: 36, l: 12 };
  let barH = $derived(height - pad.t - pad.b);
</script>

<div class="panel p-3">
  {#if title}
    <h3 class="mb-2 mt-0 text-sm font-semibold tracking-wide">{title}</h3>
  {/if}
  {#if !entries.length}
    <p class="muted m-0 text-sm">No data yet</p>
  {:else}
    <svg viewBox={`0 0 ${width} ${height}`} class="h-auto w-full" role="img" aria-label={title || 'bar chart'}>
      {#each entries as [label, value], i}
        {@const w = ((width - pad.l - pad.r) * value) / total}
        {@const x = pad.l + entries.slice(0, i).reduce((s, [, v]) => s + ((width - pad.l - pad.r) * v) / total, 0)}
        <rect x={x} y={pad.t} width={Math.max(2, w)} height={barH} fill={colors[i % colors.length]} rx="3" />
        <text x={x + w / 2} y={height - 10} text-anchor="middle" fill="var(--muted)" font-size="10">
          {label} ({value.toFixed(value >= 10 ? 0 : 1)})
        </text>
      {/each}
    </svg>
  {/if}
</div>
