<script>
  let {
    title = '',
    data = [],
    height = 160,
    color = 'var(--accent)',
    yLabel = '',
  } = $props();

  let hovered = $state(null);
  const pad = { t: 16, r: 12, b: 28, l: 40 };
  const width = 640;

  let points = $derived.by(() => {
    const rows = (data || []).filter((d) => d.y != null && Number.isFinite(Number(d.y)));
    if (!rows.length) return [];
    const xs = rows.map((d) => Number(d.x));
    const ys = rows.map((d) => Number(d.y));
    const xmin = Math.min(...xs);
    const xmax = Math.max(...xs);
    const ymin = Math.min(...ys);
    const ymax = Math.max(...ys);
    const xspan = Math.max(1e-9, xmax - xmin);
    const yspan = Math.max(1e-9, ymax - ymin);
    const innerW = width - pad.l - pad.r;
    const innerH = height - pad.t - pad.b;
    return rows.map((d, i) => {
      const x = pad.l + ((Number(d.x) - xmin) / xspan) * innerW;
      const y = pad.t + (1 - (Number(d.y) - ymin) / yspan) * innerH;
      return { ...d, px: x, py: y, i };
    });
  });

  let pathD = $derived.by(() => {
    if (points.length < 2) return '';
    return points.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.px.toFixed(1)},${p.py.toFixed(1)}`).join(' ');
  });

  let yTicks = $derived.by(() => {
    const rows = points;
    if (!rows.length) return [];
    const ys = rows.map((d) => Number(d.y));
    const ymin = Math.min(...ys);
    const ymax = Math.max(...ys);
    const ticks = [ymin, (ymin + ymax) / 2, ymax];
    const innerH = height - pad.t - pad.b;
    return ticks.map((v) => ({
      v,
      y: pad.t + (1 - (v - ymin) / Math.max(1e-9, ymax - ymin)) * innerH,
    }));
  });
</script>

<div class="panel p-3">
  {#if title}
    <div class="mb-2 flex items-baseline justify-between gap-3">
      <h3 class="m-0 text-sm font-semibold tracking-wide">{title}</h3>
      {#if yLabel}<span class="muted text-xs">{yLabel}</span>{/if}
    </div>
  {/if}
  {#if !points.length}
    <p class="muted m-0 text-sm">No data yet</p>
  {:else}
    <svg viewBox={`0 0 ${width} ${height}`} class="h-auto w-full" role="img" aria-label={title || 'line chart'}>
      {#each yTicks as t}
        <line x1={pad.l} x2={width - pad.r} y1={t.y} y2={t.y} stroke="var(--border)" stroke-width="1" />
        <text x={pad.l - 6} y={t.y + 3} text-anchor="end" fill="var(--muted)" font-size="10">
          {Number(t.v).toFixed(2)}
        </text>
      {/each}
      {#if pathD}
        <path d={pathD} fill="none" stroke={color} stroke-width="2" />
      {/if}
      {#each points as p}
        <circle
          cx={p.px}
          cy={p.py}
          r={hovered === p.i ? 4.5 : 3}
          fill={color}
          opacity={hovered === null || hovered === p.i ? 1 : 0.35}
          role="img"
          onmouseenter={() => (hovered = p.i)}
          onmouseleave={() => (hovered = null)}
        />
      {/each}
      {#if hovered != null && points[hovered]}
        <text
          x={points[hovered].px}
          y={points[hovered].py - 8}
          text-anchor="middle"
          fill="var(--text)"
          font-size="11"
        >
          {points[hovered].rollout_id || points[hovered].x}: {Number(points[hovered].y).toFixed(3)}
        </text>
      {/if}
    </svg>
  {/if}
</div>
